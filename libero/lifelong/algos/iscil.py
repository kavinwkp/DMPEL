import collections
import time
import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader, RandomSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.cluster import KMeans

from libero.lifelong.algos.base import Sequential
from libero.lifelong.datasets import TruncatedSequenceDataset
from libero.lifelong.utils import *
from libero.lifelong.metric import *
from libero.lifelong.models import *
from libero.lifelong.models.modules.adapter import *

from torch.utils.tensorboard import SummaryWriter

class ISCIL(Sequential):
    """
    TAIL method
    """

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        
    def frozen_only_observe(self, data):
        data = self.map_tensor_to_device(data)
        with torch.no_grad():
            with amp.autocast('cuda', dtype=torch.float16):
                data = self.policy.preprocess_input(data, train_mode=True, augmentation=False)
                state = self.policy.forward(data, frozen_only=True)
        return state
    
    def start_task(self, task):
        """
        What the algorithm does at the beginning of learning each lifelong task.
        """
        self.current_task = task

        # initialize the optimizer and scheduler
        self.optimizer = eval(self.cfg.train.optimizer.name)(
            self.policy.parameters(), **self.cfg.train.optimizer.kwargs
        )
        
        self.scaler = amp.GradScaler()
        self.summary_writer = SummaryWriter(log_dir=self.experiment_dir+'/tblog/'+str(task))
        self.policy.set_task_id(task)

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

        self.scheduler = None
        try:
            if self.cfg.train.scheduler is not None:
                self.cfg.train.scheduler.kwargs['epochs'] = self.cfg.train.n_epochs
                self.scheduler = eval(self.cfg.train.scheduler.name)(
                    self.optimizer,
                    steps_per_epoch = len(train_dataloader),
                    **self.cfg.train.scheduler.kwargs,
                )
        except:
            pass

        if not self.cfg.use_ddp or int(os.environ["RANK"]) == 0:
            successes = []
            losses = []
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
        
        # 1. split query by skill(shilluettte) and initialize the modulator
        # 1-2. match the query by existing model( like evaluation = argmax sim(q,k))
            # exception-first init (if first phase, hard initialization)
        # 2. evaluate the query by existing model( a ~ pi(a|s,d_id) )
        # 3. dataset['pred_actions'] and action-prototype distribution matched by key
        # 4. if novel, then expand the skill group(new context) - novel key is fixed
            # if seen context, then do not modify the key

        # initialize prototype
        state_list = []
        idx_list = []
        for (idx, data) in enumerate(train_dataloader):
            state, idx = self.frozen_only_observe(data)
            state_list.append(state)
            idx_list.append(idx)
    
        novel_contexts = torch.cat(state_list, dim=0).cpu().numpy()
        idx_cat = torch.cat(idx_list, dim=0).int().cpu().numpy().flatten()
        prototype_bases = KMeans(n_clusters=self.cfg.policy.base_per_proto, n_init=10).fit(novel_contexts).cluster_centers_
        self.policy.multifacet_prototype[task_id] = torch.from_numpy(prototype_bases)

        # initialize the adapter for novel context (except for the first task)
        if task_id > 0:
            # initialize the adapter for the new task
            idx_appears_most = np.bincount(idx_cat).argmax()
            for i, layer in enumerate(self.policy.image_encoder_spatial.model.blocks):
                if isinstance(layer.attn.qkv, L2MLoRAqkv):
                    layer.attn.qkv.A_q_pool.data[task_id] = layer.attn.qkv.A_q_pool.data[idx_appears_most]
                    layer.attn.qkv.A_v_pool.data[task_id] = layer.attn.qkv.A_v_pool.data[idx_appears_most]
                    layer.attn.qkv.B_q_pool.data[task_id] = layer.attn.qkv.B_q_pool.data[idx_appears_most]
                    layer.attn.qkv.B_v_pool.data[task_id] = layer.attn.qkv.B_v_pool.data[idx_appears_most]
                    layer.attn.qkv.freeze_expert_in_pool(task_id-1)
            for i, layer in enumerate(self.policy.language_encoder_spatial.model.encoder.layers):
                if isinstance(layer.self_attn.q_proj, L2MLoRA):
                    layer.self_attn.q_proj.A_pool.data[task_id] = layer.self_attn.q_proj.A_pool.data[idx_appears_most]
                    layer.self_attn.q_proj.B_pool.data[task_id] = layer.self_attn.q_proj.B_pool.data[idx_appears_most]
                    layer.self_attn.q_proj.freeze_expert_in_pool(task_id-1)
                if isinstance(layer.self_attn.v_proj, L2MLoRA):
                    layer.self_attn.v_proj.A_pool.data[task_id] = layer.self_attn.v_proj.A_pool.data[idx_appears_most]
                    layer.self_attn.v_proj.B_pool.data[task_id] = layer.self_attn.v_proj.B_pool.data[idx_appears_most]
                    layer.self_attn.v_proj.freeze_expert_in_pool(task_id-1)
            for i, layer in enumerate(self.policy.temporal_transformer.layers):
                if isinstance(layer[1].qkv, L2MLoRAqkv):
                    layer[1].qkv.A_q_pool.data[task_id] = layer[1].qkv.A_q_pool.data[idx_appears_most]
                    layer[1].qkv.A_v_pool.data[task_id] = layer[1].qkv.A_v_pool.data[idx_appears_most]
                    layer[1].qkv.B_q_pool.data[task_id] = layer[1].qkv.B_q_pool.data[idx_appears_most]
                    layer[1].qkv.B_v_pool.data[task_id] = layer[1].qkv.B_v_pool.data[idx_appears_most]
                    layer[1].qkv.freeze_expert_in_pool(task_id-1)
            if not self.policy.lora_only:
                for i, encoder in enumerate(self.policy.extra_encoder.encoders):
                    for j, linear in enumerate(encoder):
                        if isinstance(linear, L2MLinear):
                            linear.W_pool.data[task_id] = linear.W_pool.data[idx_appears_most]
                            linear.B_pool.data[task_id] = linear.B_pool.data[idx_appears_most]
                            linear.freeze_expert_in_pool(task_id-1)
                for i, linear in enumerate(self.policy.fusion_module):
                    if isinstance(linear, L2MLinear):
                        linear.W_pool.data[task_id] = linear.W_pool.data[idx_appears_most]
                        linear.B_pool.data[task_id] = linear.B_pool.data[idx_appears_most]
                        linear.freeze_expert_in_pool(task_id-1)
                for i, linear in enumerate(self.policy.policy_head.share):
                    if isinstance(linear, L2MLinear):
                        linear.W_pool.data[task_id] = linear.W_pool.data[idx_appears_most]
                        linear.B_pool.data[task_id] = linear.B_pool.data[idx_appears_most]
                        linear.freeze_expert_in_pool(task_id-1)
                if isinstance(self.policy.policy_head.mean_layer, L2MLinear):
                    self.policy.policy_head.mean_layer.W_pool.data[task_id] = \
                        self.policy.policy_head.mean_layer.W_pool.data[idx_appears_most]
                    self.policy.policy_head.mean_layer.B_pool.data[task_id] = \
                        self.policy.policy_head.mean_layer.B_pool.data[idx_appears_most]
                    self.policy.policy_head.mean_layer.freeze_expert_in_pool(task_id-1)
                if isinstance(self.policy.policy_head.logstd_layer, L2MLinear):
                    self.policy.policy_head.logstd_layer.W_pool.data[task_id] = \
                        self.policy.policy_head.logstd_layer.W_pool.data[idx_appears_most]
                    self.policy.policy_head.logstd_layer.B_pool.data[task_id] = \
                        self.policy.policy_head.logstd_layer.B_pool.data[idx_appears_most]
                    self.policy.policy_head.logstd_layer.freeze_expert_in_pool(task_id-1)
                if isinstance(self.policy.policy_head.logits_layer, L2MLinear):
                    self.policy.policy_head.logits_layer.W_pool.data[task_id] = \
                        self.policy.policy_head.logits_layer.W_pool.data[idx_appears_most]
                    self.policy.policy_head.logits_layer.B_pool.data[task_id] = \
                        self.policy.policy_head.logits_layer.B_pool.data[idx_appears_most]
                    self.policy.policy_head.logits_layer.freeze_expert_in_pool(task_id-1)
        else:
            if not self.policy.lora_only:
                for i, encoder in enumerate(self.policy.extra_encoder.encoders):
                    for j, linear in enumerate(encoder):
                        if isinstance(linear, L2MLinear):
                            linear.W_pool.data[task_id] = linear.weight.T
                            linear.B_pool.data[task_id] = linear.bias
                            linear.freeze_expert_in_pool(task_id-1)
                for i, linear in enumerate(self.policy.fusion_module):
                    if isinstance(linear, L2MLinear):
                        linear.W_pool.data[task_id] = linear.weight.T
                        linear.B_pool.data[task_id] = linear.bias
                        linear.freeze_expert_in_pool(task_id-1)
                for i, linear in enumerate(self.policy.policy_head.share):
                    if isinstance(linear, L2MLinear):
                        linear.W_pool.data[task_id] = linear.weight.T
                        linear.B_pool.data[task_id] = linear.bias
                        linear.freeze_expert_in_pool(task_id-1)
                if isinstance(self.policy.policy_head.mean_layer, L2MLinear):
                    self.policy.policy_head.mean_layer.W_pool.data[task_id] = \
                        self.policy.policy_head.mean_layer.weight.T
                    self.policy.policy_head.mean_layer.B_pool.data[task_id] = \
                        self.policy.policy_head.mean_layer.bias
                    self.policy.policy_head.mean_layer.freeze_expert_in_pool(task_id-1)
                if isinstance(self.policy.policy_head.logstd_layer, L2MLinear):
                    self.policy.policy_head.logstd_layer.W_pool.data[task_id] = \
                        self.policy.policy_head.logstd_layer.weight.T
                    self.policy.policy_head.logstd_layer.B_pool.data[task_id] = \
                        self.policy.policy_head.logstd_layer.bias
                    self.policy.policy_head.logstd_layer.freeze_expert_in_pool(task_id-1)
                if isinstance(self.policy.policy_head.logits_layer, L2MLinear):
                    self.policy.policy_head.logits_layer.W_pool.data[task_id] = \
                        self.policy.policy_head.logits_layer.weight.T
                    self.policy.policy_head.logits_layer.B_pool.data[task_id] = \
                        self.policy.policy_head.logits_layer.bias
        
        # start training
        for epoch in range(0, self.cfg.train.n_epochs + 1):

            t0 = time.time()

            if self.cfg.use_ddp:
                train_dataloader.sampler.set_epoch(epoch)

            if epoch > 0:  # update
                self.policy.train()
                training_loss = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    loss = self.observe(data)
                    training_loss += loss
                    if self.scheduler is not None:
                        self.scheduler.step()
            else:  # just evaluate the zero-shot performance on 0-th epoch
                training_loss = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    loss = self.eval_observe(data)
                    training_loss += loss
                    
            t1 = time.time()

            if torch.cuda.is_available():
                MB = 1024.0 * 1024.0
                peak_memory = (torch.cuda.max_memory_allocated() / MB) / 1000 
            else:
                peak_memory = 0

            if self.cfg.use_ddp:
                training_loss = torch.as_tensor(training_loss, device=torch.device("cuda:"+os.environ["RANK"]))
                peak_memory = torch.as_tensor(peak_memory, device=torch.device("cuda:"+os.environ["RANK"]))
                dataloader_len = torch.as_tensor(len(train_dataloader), device=torch.device("cuda:"+os.environ["RANK"]))
                
                training_loss_gather_list = [torch.zeros_like(training_loss) for _ in range(dist.get_world_size())] if int(os.environ["RANK"]) == 0 else None
                peak_memory_gather_list = [torch.zeros_like(peak_memory) for _ in range(dist.get_world_size())] if int(os.environ["RANK"]) == 0 else None
                dataloader_len_gather_list = [torch.zeros_like(dataloader_len) for _ in range(dist.get_world_size())] if int(os.environ["RANK"]) == 0 else None

                dist.gather(training_loss, training_loss_gather_list, dst=0)
                dist.gather(peak_memory, peak_memory_gather_list, dst=0)
                dist.gather(dataloader_len, dataloader_len_gather_list, dst=0)
                
                if int(os.environ["RANK"]) == 0:
                    training_loss = sum(training_loss_gather_list).item() / sum(dataloader_len_gather_list).item()
                    peak_memory = sum(peak_memory_gather_list).item()
                    print(
                        f'[info] # Batch: {sum(dataloader_len_gather_list).item()} | Epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1-t0)/60:4.2f} | '
                        f'Memory utilization: %.3f GB' % peak_memory
                    )
            else:
                training_loss /= len(train_dataloader)
                print(
                    f'[info] # Batch: {len(train_dataloader)} | Epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1-t0)/60:4.2f} | '
                    f'Memory utilization: %.3f GB' % peak_memory
                )
            
            if (not self.cfg.use_ddp) or (int(os.environ["RANK"]) == 0):
                self.summary_writer.add_scalar("train_loss", training_loss, epoch)
                # for name, param in self.policy.named_parameters():
                    # if not param.requires_grad:
                        # self.summary_writer.add_histogram(f'value/{name}', param, epoch)
                        # self.summary_writer.add_histogram(f'grad/{name}.grad', param.grad, epoch)
                        # self.summary_writer.add_histogram(f'grad/{name}.grad', param.grad, epoch)

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
                losses.append(training_loss)
                successes.append(success_rate)
                
                self.summary_writer.add_scalar("success_rate", success_rate, epoch)

                if prev_success_rate < success_rate:
                    torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg, learnable_only=True)
                    prev_success_rate = success_rate
                    idx_at_best_succ = len(losses) - 1

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
            # print(f'[info] {msg}')

        # end learning the current task, some algorithms need post-processing
        self.end_task(dataset, task_id, benchmark)
        
        if (not self.cfg.use_ddp or int(os.environ["RANK"]) == 0) and (not self.cfg.debug_no_eval):
            # return the metrics regarding forward transfer
            losses = np.array(losses)
            successes = np.array(successes)
            epochs = np.array(epochs)
            peak_memories = np.array(peak_memories)
            auc_checkpoint_name = os.path.join(
                self.experiment_dir, f"task{task_id}_auc.log"
            )
            torch.save(
                {   "epochs": epochs,
                    "success": successes,
                    "loss": losses,
                    "peak_memories": peak_memories,
                },
                auc_checkpoint_name,
            )

            # pretend that the agent stops learning once it reaches the peak performance
            losses[idx_at_best_succ:] = losses[idx_at_best_succ]
            successes[idx_at_best_succ:] = successes[idx_at_best_succ]
            return successes.sum() / cumulated_counter, losses.sum() / cumulated_counter
        else:
            return 0.0, 0.0