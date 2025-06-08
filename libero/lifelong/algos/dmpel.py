import os
import time

import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.utils.data import DataLoader, RandomSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.optim import AdamW
from sklearn.cluster import KMeans

from libero.lifelong.algos.base import Sequential
from libero.lifelong.datasets import TruncatedSequenceDataset
from libero.lifelong.metric import *
from libero.lifelong.models import *
from libero.lifelong.utils import *

from torch.utils.tensorboard import SummaryWriter

class CustomDDP(DDP):
    """
    The default DistributedDataParallel enforces access to class the module attributes via self.module. 
    This is impractical for our use case, as we need to access certain module access throughout. 
    We override the __getattr__ method to allow access to the module attributes directly.
    
    For example: 
    ```
        # default behaviour
        model = OnlineDecisionTransformerModel()
        model = DistributedDataParallel(model)
        model.module.some_attribute
        
        # custom behaviour using this class
        model = OnlineDecisionTransformerModel()
        model = CustomDDP(model)
        model.some_attribute
        
    ```        
    Shoudl not cause any inconsistencies: 
    https://discuss.pytorch.org/t/access-to-attributes-of-model-wrapped-in-ddp/130572
    
    """
    
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class MoeAttnReplayDataset(Dataset):
    def __init__(self, task_id, exp_dir, cfg, pool_size):
        self.moe_query_in_list = []
        self.moe_query_emb_list = []
        self.topk_attn_norm_list = []
        for i in range(task_id + 1):
            moe_attn_recall_data = torch.load(os.path.join(exp_dir, f"task{i}_moe_attn_recall_query.pth"))
            moe_query_in = [torch.cat((moe_attn_recall_data['moe_query_in_txt'], query_img), dim=-1) 
                            for query_img in list(torch.unbind(moe_attn_recall_data['moe_query_in_img'], dim=0))]
            num_samples = len(moe_query_in)
            moe_query_in = torch.stack(moe_query_in, dim=0)
            moe_attn_recall_data_attn = torch.load(os.path.join(exp_dir, f"task{i}_moe_attn_recall_attn.pth"))
            topk_idx = moe_attn_recall_data_attn['topk_idx'].long()
            topk_attn_norm = moe_attn_recall_data_attn['topk_attn_norm']
            if cfg.policy.router_coeff_seperate:
                topk_attn_norm_multi = topk_attn_norm.view(num_samples, 6, -1)
                topk_idx_multi = topk_idx.view(num_samples, 6, -1)
                attn_vector = torch.zeros((num_samples, 6, pool_size), device=topk_attn_norm.device)
                for i in range(6):
                    attn_vector[:, i, :].scatter_(1, topk_idx_multi[:, i, :], topk_attn_norm_multi[:, i, :])
                attn_vector.view(num_samples, -1)
            else:
                attn_vector = torch.zeros((num_samples, pool_size), device=topk_attn_norm.device)
                attn_vector.scatter_(1, topk_idx, topk_attn_norm)
            self.moe_query_in_list.append(moe_query_in)
            self.topk_attn_norm_list.append(attn_vector)
        self.moe_query_in_list = torch.cat(self.moe_query_in_list, dim=0)
        self.topk_attn_norm_list = torch.cat(self.topk_attn_norm_list, dim=0)

    def __len__(self):
        return len(self.moe_query_in_list)
    
    def __getitem__(self, idx):
        moe_query_in = self.moe_query_in_list[idx]
        topk_attn_norm = self.topk_attn_norm_list[idx]
        return moe_query_in, topk_attn_norm

class DMPEL(Sequential):
    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)

    def start_task(self, task):
        self.current_task = task
        self.scaler = amp.GradScaler()
        self.summary_writer = SummaryWriter(log_dir=self.experiment_dir+'/tblog/'+str(task))
    
    def end_task(self, dataset, task_id, benchmark, env=None):
        if self.cfg.lifelong.moe_attn_recall_epochs > 0 and task_id > 0:
            print("[info] start recalling attention...")
            moe_attn_replay_dataset = MoeAttnReplayDataset(task_id, self.experiment_dir, self.cfg, self.policy.moe_router.pool_size)
            self.recall_optimizer = AdamW(self.policy.moe_router.parameters())
            self.recall_scaler = amp.GradScaler()
            multiprocessing.set_start_method("fork", force=True)
            train_dataloader = DataLoader(
                moe_attn_replay_dataset,
                batch_size=self.cfg.train.batch_size,
                # num_workers=self.cfg.train.num_workers,
                sampler=RandomSampler(moe_attn_replay_dataset),
                # persistent_workers=True,
            )

            best_recall_loss = float('inf')
            for epoch in range(self.cfg.lifelong.moe_attn_recall_epochs):
                recall_loss_avg = 0.0
                attn_list = []
                for (idx, data) in enumerate(train_dataloader):
                    data = self.map_tensor_to_device(data)
                    self.recall_optimizer.zero_grad()
                    with amp.autocast('cuda', dtype=torch.float32):
                        loss, attn_vector = self.policy.recall_moe_attention(data[0], data[1])
                        self.recall_scaler.scale(loss).backward()
                    if self.cfg.train.grad_clip is not None:
                        self.recall_scaler.unscale_(self.recall_optimizer)
                        grad_norm = nn.utils.clip_grad_norm_(
                            self.policy.moe_router.parameters(), self.cfg.train.grad_clip
                        )
                    self.recall_scaler.step(self.recall_optimizer)
                    self.recall_scaler.update()
                    recall_loss_avg += loss.item()
                    attn_list.append(attn_vector)
                
                recall_loss_avg /= len(train_dataloader)
                if recall_loss_avg < best_recall_loss:
                    best_recall_loss = recall_loss_avg
                    torch_save_model(self.policy.moe_router, os.path.join(self.experiment_dir, f"task{task_id}_moe_router.pth"), cfg=self.cfg)
                print(f'[info] Epoch: {epoch:3d} | recall loss: {recall_loss_avg:5.4f}')
                self.summary_writer.add_scalar("recall/recall_loss", recall_loss_avg, epoch)
            msg = self.policy.moe_router.load_state_dict(torch_load_model(os.path.join(self.experiment_dir, f"task{task_id}_moe_router.pth"))[0], strict=False)
            print(f'[info] {msg}')
            print("[info] end recalling attention...")
    
    def observe(self, data):
        """
        How the algorithm learns on each data point.
        """
        data = self.map_tensor_to_device(data)
        self.optimizer.zero_grad()
        # torch.autograd.set_detect_anomaly(True)  # detect anomaly       
        with amp.autocast('cuda', dtype=torch.float16):
            bc_loss = self.policy.compute_loss(data)
        # with torch.autograd.detect_anomaly():
        self.scaler.scale(self.loss_scale * bc_loss).backward()
        if self.cfg.train.grad_clip is not None:
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip
            )
            # print(f'[info] Gradient norm: {grad_norm}')
            # for name, tensor in self.policy.named_parameters():
            #     if tensor.grad is not None:
            #         tensor_grad_norm = torch.norm(tensor.grad)
            #         print(f'[info] {name} grad norm: {tensor_grad_norm}')
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return bc_loss.item()
    
    def eval_observe(self, data):
        data = self.map_tensor_to_device(data)
        with torch.no_grad():
            with amp.autocast('cuda', dtype=torch.float16):
                bc_loss = self.policy.compute_loss(data)
        return bc_loss.item()
    
    def save_attn_observe(self, data):
        data = self.map_tensor_to_device(data)
        with torch.no_grad():
            with amp.autocast('cuda', dtype=torch.float16):
                data = self.policy.preprocess_input(data, train_mode=True, augmentation=False)
                _, moe_query_in = self.policy.context_encode(data)
                topk_idx, topk_attn_norm = self.policy.infer_lora(moe_query_in, mode='save_attn')
        return moe_query_in, topk_idx, topk_attn_norm
    
    def learn_one_task(self, dataset, task_id, benchmark, result_summary):

        self.start_task(task_id)

        # recover the corresponding manipulation task ids
        gsz = self.cfg.data.task_group_size
        manip_task_ids = list(range(task_id * gsz, (task_id + 1) * gsz))

        model_checkpoint_name = os.path.join(
            self.experiment_dir, f"task{task_id}_model.pth"
        )
        multiprocessing.set_start_method("fork", force=True)
        if self.cfg.use_ddp:
            torch.distributed.barrier()
            train_dataloader = DataLoader(
                    dataset,
                    batch_size=self.cfg.train.batch_size,
                    num_workers=self.cfg.train.num_workers,
                    shuffle=False,
                    sampler=DistributedSampler(dataset),
                    persistent_workers=True,
            )
        else:
            train_dataloader = DataLoader(
                    dataset,
                    batch_size=self.cfg.train.batch_size,
                    num_workers=self.cfg.train.num_workers,
                    sampler=RandomSampler(dataset),
                    persistent_workers=True,
                )

        if not self.cfg.use_ddp or int(os.environ["RANK"]) == 0:
            successes = []
            training_losses = []
            epochs = []
            peak_memories = []
            
            # for evaluate how fast the agent learns on current task, this corresponds
            # to the area under success rate curve on the new task.
            cumulated_counter = 0.0
            idx_at_best_succ = 0

            prev_success_rate = -1.0
            # best_state_dict = self.policy.state_dict()  # currently save the best model

        task = benchmark.get_task(task_id)
        task_emb = benchmark.get_task_emb(task_id)

        self.policy.add_new_and_freeze_previous(self.cfg.policy.ll_expert_per_task)
        
        self.optimizer = eval(self.cfg.train.optimizer.name)(
                    self.policy.parameters(),
                    **self.cfg.train.optimizer.kwargs,
                )

        self.scheduler = None
        try:
            if self.cfg.train.scheduler is not None:
                self.router_lr_scale = self.cfg.policy.router_lr_scale
                self.base_lr = self.cfg.train.optimizer.kwargs['lr']
                self.cfg.train.scheduler.kwargs['max_lr'] = [self.router_lr_scale * self.base_lr, self.base_lr]
                self.cfg.train.scheduler.kwargs['epochs'] = self.cfg.train.n_epochs
                router_list = []
                other_list = []
                for tensor_name, tensor in self.policy.named_parameters():
                    if "moe_router" in tensor_name:
                        router_list.append(tensor)
                    else:
                        other_list.append(tensor)
                self.optimizer = eval(self.cfg.train.optimizer.name)(
                    [{'params': router_list, 'lr': self.router_lr_scale * self.base_lr}, 
                    {'params': other_list}],
                    **self.cfg.train.optimizer.kwargs
                )
        except:
            pass    
        
        # start training
        for epoch in range(0, self.cfg.train.n_epochs + 1):

            t0 = time.time()

            if self.cfg.use_ddp:
                train_dataloader.sampler.set_epoch(epoch)

            if epoch > 0:  # update
                self.policy.train()
                training_loss_avg = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    loss = self.observe(data)
                    training_loss_avg += loss
                    if self.scheduler is not None:
                        self.scheduler.step()
            else:  # just evaluate the zero-shot performance on 0-th epoch
                training_loss_avg = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    loss = self.eval_observe(data)
                    training_loss_avg += loss
            t1 = time.time()

            if torch.cuda.is_available():
                MB = 1024.0 * 1024.0
                peak_memory = (torch.cuda.max_memory_allocated() / MB) / 1000 
            else:
                peak_memory = 0

            if self.cfg.use_ddp:
                training_loss_avg = torch.as_tensor(training_loss_avg, device=torch.device("cuda:"+os.environ["RANK"]))
                peak_memory = torch.as_tensor(peak_memory, device=torch.device("cuda:"+os.environ["RANK"]))
                dataloader_len = torch.as_tensor(len(train_dataloader), device=torch.device("cuda:"+os.environ["RANK"]))
                
                training_loss_gather_list = [torch.zeros_like(training_loss_avg) for _ in range(dist.get_world_size())] if int(os.environ["RANK"]) == 0 else None
                peak_memory_gather_list = [torch.zeros_like(peak_memory) for _ in range(dist.get_world_size())] if int(os.environ["RANK"]) == 0 else None
                dataloader_len_gather_list = [torch.zeros_like(dataloader_len) for _ in range(dist.get_world_size())] if int(os.environ["RANK"]) == 0 else None

                dist.gather(training_loss_avg, training_loss_gather_list, dst=0)
                dist.gather(peak_memory, peak_memory_gather_list, dst=0)
                dist.gather(dataloader_len, dataloader_len_gather_list, dst=0)
                
                if int(os.environ["RANK"]) == 0:
                    training_loss = sum(training_loss_gather_list).item() / sum(dataloader_len_gather_list).item()
                    peak_memory = sum(peak_memory_gather_list).item()
                    print(
                        f'[info] # Batch: {sum(dataloader_len_gather_list).item()} | Epoch: {epoch:3d} | '
                        f'train loss: {training_loss:5.2f} | '
                        f'time: {(t1-t0)/60:4.2f} | Memory utilization: %.3f GB' % peak_memory
                    )
            else:
                training_loss_avg /= len(train_dataloader)
                print(
                    f'[info] # Batch: {len(train_dataloader)} | Epoch: {epoch:3d} | '
                    f'train loss: {training_loss_avg:5.2f} | '
                    f'time: {(t1-t0)/60:4.2f} | Memory utilization: %.3f GB' % peak_memory
                )
            
            if (not self.cfg.use_ddp) or (int(os.environ["RANK"]) == 0):
                self.summary_writer.add_scalar("bc/train_loss", training_loss_avg, epoch)
                self.summary_writer.add_scalar("bc/peak_memory", peak_memory, epoch)
                
            if epoch % self.cfg.eval.eval_every == 0 and (not self.cfg.use_ddp or int(os.environ["RANK"]) == 0) and (not self.cfg.debug_no_eval):  # evaluate BC loss
                # every eval_every epoch, we evaluate the agent on the current task,
                # then we pick the best performant agent on the current task as
                # if it stops learning after that specific epoch. So the stopping
                # criterion for learning a new task is achieving the peak performance
                # on the new task. Future work can explore how to decide this stopping
                # epoch by also considering the agent's performance on old tasks.
                
                t0 = time.time()

                task_str = f"k{task_id}_e{epoch//self.cfg.eval.eval_every}"
                sim_states = (
                    result_summary[task_str] if self.cfg.eval.save_sim_states else None
                )
                
                success_rate = evaluate_one_task_success(
                    cfg=self.cfg,
                    algo=self,
                    task=task,
                    task_emb=task_emb,
                    task_id=task_id,
                    sim_states=sim_states,
                    task_str="",
                )
                
                epochs.append(epoch)
                peak_memories.append(peak_memory)
                training_losses.append(training_loss_avg)
                successes.append(success_rate)
                
                self.summary_writer.add_scalar("success_rate", success_rate, epoch)

                if prev_success_rate < success_rate:
                    torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg, learnable_only=True)
                    prev_success_rate = success_rate
                    idx_at_best_succ = len(training_losses) - 1

                    cumulated_counter += 1.0
                    ci = confidence_interval(success_rate, self.cfg.eval.n_eval)
                    tmp_successes = np.array(successes)
                    tmp_successes[idx_at_best_succ:] = successes[idx_at_best_succ]
                
                t1 = time.time()
                print(
                    f"[info] Epoch: {epoch:3d} | succ: {success_rate:4.2f} ± {ci:4.2f} | best succ: {prev_success_rate} "
                    + f"| succ. AoC {tmp_successes.sum()/cumulated_counter:4.2f} | time: {(t1-t0)/60:4.2f}",
                    flush=True,
                )

            if self.cfg.use_ddp:
                torch.distributed.barrier()
        
        # load the best performance agent on the current task
        if (not self.cfg.debug_no_eval):
            msg = self.policy.load_state_dict(torch_load_model(model_checkpoint_name)[0], strict=False)
            print(f'[info] {msg}')
        
        # end learning the current task, some algorithms need post-processing
        if self.cfg.lifelong.moe_attn_recall_epochs > 0:
            moe_query_in_img_list = []
            moe_query_list = []
            topk_idx_list = []
            topk_attn_norm_list = []
            for (idx, data) in enumerate(train_dataloader):
                moe_query_in, topk_idx, topk_attn_norm = self.save_attn_observe(data)
                moe_query_in_txt = moe_query_in[0, :self.policy.language_embed_dim]
                moe_query_in_img = moe_query_in[..., self.policy.language_embed_dim:]
                moe_query_in_img_list.append(moe_query_in_img)
                topk_idx_list.append(topk_idx)
                topk_attn_norm_list.append(topk_attn_norm)
            
            moe_query_in_img_list = torch.cat(moe_query_in_img_list, dim=0)
            
            topk_idx_list = torch.cat(topk_idx_list, dim=0).int()
            topk_attn_norm_list = torch.cat(topk_attn_norm_list, dim=0)
            total_sample_num = len(moe_query_in_img_list)
            saved_sample_num = int(self.cfg.lifelong.moe_attn_recall_sample_ratio * total_sample_num)
            rand_indices = torch.randperm(total_sample_num, device=moe_query_in_img_list.device)[:saved_sample_num]
            torch.save({
                'moe_query_in_txt': moe_query_in_txt,
                'moe_query_in_img': moe_query_in_img_list.index_select(dim=0, index=rand_indices), 
                },
                os.path.join(self.experiment_dir, f"task{task_id}_moe_attn_recall_query.pth")
            )
            torch.save({
                'topk_idx': topk_idx_list.index_select(dim=0, index=rand_indices), 
                'topk_attn_norm': topk_attn_norm_list.index_select(dim=0, index=rand_indices),
                },
                os.path.join(self.experiment_dir, f"task{task_id}_moe_attn_recall_attn.pth")
            )

            if self.cfg.policy.router_coeff_seperate:
                topk_attn_norm_multi = topk_attn_norm_list.view(topk_attn_norm_list.shape[0], 6, -1)
                topk_idx_multi = topk_idx_list.view(topk_attn_norm_list.shape[0], 6, -1)
                expert_stats = {}
                for i, key in zip(range(6), ['img', 'txt', 'extra', 'fusion', 'tem', 'head']):
                    flat_indices = torch.flatten(topk_idx_multi[:, i, :]).long()
                    flat_attn = torch.flatten(topk_attn_norm_multi[:, i, :]).float()
                    frequency = torch.bincount(flat_indices, minlength=self.policy.pool_size)
                    total_attn = torch.bincount(flat_indices, weights=flat_attn, minlength=self.policy.pool_size)
                    avg_attn = torch.zeros_like(total_attn, dtype=torch.float)
                    non_zero_mask = frequency > 0
                    avg_attn[non_zero_mask] = total_attn[non_zero_mask] / frequency[non_zero_mask]

                    expert_stats[key] = {
                            str(expert_id): {
                                "frequency": frequency[expert_id].item(),
                                "total_attention": total_attn[expert_id].item(),
                                "average_attention": avg_attn[expert_id].item() if non_zero_mask[expert_id] else 0.0
                            }
                            for expert_id in range(self.policy.pool_size)
                        }
                    
                    print(f"{key}")
                    print(f"{'Expert ID':<10} {'Frequency':<10} {'Total Attn':<12} {'Avg Attn':<10}")
                    for expert_id, stats in expert_stats[key].items():
                        print(f"{expert_id:<10} {stats['frequency']:<10} {stats['total_attention']:<12.4f} {stats['average_attention']:<10.4f}")
                torch.save(
                    expert_stats,
                    os.path.join(self.experiment_dir, f"task{task_id}_expert_stats.pth")
                )    
            else:
                flat_indices = topk_idx_list.view(-1).long()
                flat_attn = topk_attn_norm_list.view(-1).float()

                frequency = torch.bincount(flat_indices, minlength=self.policy.pool_size)
                total_attn = torch.bincount(flat_indices, weights=flat_attn, minlength=self.policy.pool_size)
                avg_attn = torch.zeros_like(total_attn, dtype=torch.float)
                non_zero_mask = frequency > 0
                avg_attn[non_zero_mask] = total_attn[non_zero_mask] / frequency[non_zero_mask]
                expert_stats = {
                    str(expert_id): {
                        "frequency": frequency[expert_id].item(),
                        "total_attention": total_attn[expert_id].item(),
                        "average_attention": avg_attn[expert_id].item() if non_zero_mask[expert_id] else 0.0
                    }
                    for expert_id in range(self.policy.pool_size)
                }
                torch.save(
                    expert_stats,
                    os.path.join(self.experiment_dir, f"task{task_id}_expert_stats.pth")
                )
                print(f"{'Expert ID':<10} {'Frequency':<10} {'Total Attn':<12} {'Avg Attn':<10}")
                for expert_id, stats in expert_stats.items():
                    print(f"{expert_id:<10} {stats['frequency']:<10} {stats['total_attention']:<12.4f} {stats['average_attention']:<10.4f}")
        
        self.end_task(dataset, task_id, benchmark)

        torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg, learnable_only=True)
        
        if (not self.cfg.use_ddp or int(os.environ["RANK"]) == 0) and (not self.cfg.debug_no_eval):
            # return the metrics regarding forward transfer
            training_losses = np.array(training_losses)
            successes = np.array(successes)
            epochs = np.array(epochs)
            peak_memories = np.array(peak_memories)
            auc_checkpoint_name = os.path.join(
                self.experiment_dir, f"task{task_id}_auc.log"
            )
            torch.save(
                {   "epochs": epochs,
                    "success": successes,
                    "training_loss": training_losses,
                    "peak_memories": peak_memories,
                },
                auc_checkpoint_name,
            )

            # pretend that the agent stops learning once it reaches the peak performance
            training_losses[idx_at_best_succ:] = training_losses[idx_at_best_succ]
            successes[idx_at_best_succ:] = successes[idx_at_best_succ]
            return successes.sum() / cumulated_counter, training_losses.sum() / cumulated_counter
        else:
            return 0.0, 0.0
