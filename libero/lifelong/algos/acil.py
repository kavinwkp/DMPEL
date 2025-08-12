import os
import time

import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch import amp
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



class ACILLearner(Sequential):
    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        self.initialized = False
        self.buffer_size = 8192

    def start_task(self, task):
        self.current_task = task
        # self.scaler = amp.GradScaler()
        self.summary_writer = SummaryWriter(log_dir=self.experiment_dir + '/tblog/' + str(task))

    @torch.no_grad()
    def end_task(self, dataset, task_id, benchmark, env=None):
        print("[info] start update acil router")
        # self.policy.init_router()


        out_features = 6 * (self.policy.moe_router.pool_size - 1)

        self.acil_router = ACIL(
            backbone_output_size=2057,
            buffer_size=8192,
            out_features=out_features,
            gamma=0.1,
            device=self.cfg.device,
            dtype=torch.double)     # TODO: don't change to float32
        print(self.acil_router)


        if task_id > 0:
            router_checkpoint_name = f"/home/kavin/Documents/GitProjects/CL/DMPEL/experiments/lifelong/acil/{benchmark.name}/run_00{task_id-1}/seed_100/acil{task_id-1}_router.pth"
            self.acil_router.load_state_dict(torch_load_model(router_checkpoint_name)[0])
            print(f'[info] load router from {router_checkpoint_name}')

        train_dataloader = DataLoader(
            dataset,
            batch_size=256,
            num_workers=0,
            sampler=RandomSampler(dataset),
        )

        t0 = time.time()

        for (idx, data) in enumerate(train_dataloader):
            data = self.map_tensor_to_device(data)
            # print(data["obs"]["agentview_rgb"].shape)
            query_in, coeff = self.policy.calc(data)    # TODO: coeff before topk
            # print(query_in.shape)       # (bs, 2057)
            # print(coeff.shape) # (bs, 6*k)

            # TODO: for incremental size
            bs = query_in.shape[0]
            coeff = coeff.reshape(bs, 6, -1)    # (bs, 6, k)
            coeff = coeff.transpose(1, 2)       # (bs, k, 6)
            Y = coeff.reshape(bs, -1)           # (bs, 6*k) e.g. [1,10,2,20,3,30] -> [1,2,3,10,20,30]

            # bs = query_in.shape[0]
            # coeff = coeff.reshape(bs, 6, -1)
            # zeros = torch.zeros((bs, 6, 10 - coeff.shape[2]), dtype=coeff.dtype, device=coeff.device)
            # Y = torch.cat((coeff, zeros), dim=-1).reshape(bs, -1)  # (bs,60)

            self.acil_router.fit(query_in, Y)

            t1 = time.time()
            print(f"[info] {idx} | Time: {(t1 - t0) / 60:4.2f}")
            # break

        model_checkpoint_name = os.path.join(self.experiment_dir, f"acil{task_id}_router.pth")
        torch_save_model(self.acil_router, model_checkpoint_name, cfg=self.cfg, learnable_only=False)

        print("[info] end recalling attention...")

    def observe(self, data):
        """
        How the algorithm learns on each data point.
        """
        data = self.map_tensor_to_device(data)
        self.optimizer.zero_grad()
        # torch.autograd.set_detect_anomaly(True)  # detect anomaly
        # with amp.autocast('cuda', dtype=torch.float16):
        bc_loss = self.policy.compute_loss(data)
        # with torch.autograd.detect_anomaly():
        # self.scaler.scale(self.loss_scale * bc_loss).backward()
        (self.loss_scale * bc_loss).backward()
        if self.cfg.train.grad_clip is not None:
            # self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip
            )
            # print(f'[info] Gradient norm: {grad_norm}')
            # for name, tensor in self.policy.named_parameters():
            #     if tensor.grad is not None:
            #         tensor_grad_norm = torch.norm(tensor.grad)
            #         print(f'[info] {name} grad norm: {tensor_grad_norm}')
        # self.scaler.step(self.optimizer)
        # self.scaler.update()
        self.optimizer.step()
        return bc_loss.item()

    def eval_observe(self, data):
        data = self.map_tensor_to_device(data)
        with torch.no_grad():
            # with amp.autocast('cuda', dtype=torch.float16):
            bc_loss = self.policy.compute_loss(data)
        return bc_loss.item()

    def save_attn_observe(self, data):
        data = self.map_tensor_to_device(data)
        with torch.no_grad():
            # with amp.autocast('cuda', dtype=torch.float16):
            data = self.policy.preprocess_input(data, train_mode=True, augmentation=False)
            _, moe_query_in = self.policy.context_encode(data)
            topk_idx, topk_attn_norm = self.policy.infer_lora(moe_query_in, mode='save_attn')
        return moe_query_in, topk_idx, topk_attn_norm

    # def learn_one_task(self, dataset, task_id, benchmark, result_summary):
    #
    #     self.start_task(task_id)
    #
    #     # recover the corresponding manipulation task ids
    #     gsz = self.cfg.data.task_group_size
    #     manip_task_ids = list(range(task_id * gsz, (task_id + 1) * gsz))
    #
    #     model_checkpoint_name = os.path.join(
    #         self.experiment_dir, f"task{task_id}_model.pth"
    #     )
    #
    #     train_dataloader = DataLoader(
    #         dataset,
    #         batch_size=self.cfg.train.batch_size,
    #         num_workers=self.cfg.train.num_workers,
    #         sampler=RandomSampler(dataset),
    #         persistent_workers=True,
    #     )
    #
    #     # if not self.cfg.use_ddp or int(os.environ["RANK"]) == 0:
    #     successes = []
    #     training_losses = []
    #     epochs = []
    #     peak_memories = []
    #
    #     # for evaluate how fast the agent learns on current task, this corresponds
    #     # to the area under success rate curve on the new task.
    #     cumulated_counter = 0.0
    #     idx_at_best_succ = 0
    #
    #     prev_success_rate = -1.0
    #     # best_state_dict = self.policy.state_dict()  # currently save the best model
    #
    #     task = benchmark.get_task(task_id)
    #     task_emb = benchmark.get_task_emb(task_id)
    #
    #     self.policy.add_new_and_freeze_previous(self.cfg.policy.ll_expert_per_task)
    #
    #
    #     self.end_task(dataset, task_id, benchmark)
    #
    #     # torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg, learnable_only=True)
    #
    #     # if (not self.cfg.use_ddp or int(os.environ["RANK"]) == 0) and (not self.cfg.debug_no_eval):
    #     # return the metrics regarding forward transfer
    #     training_losses = np.array(training_losses)
    #     successes = np.array(successes)
    #     epochs = np.array(epochs)
    #     peak_memories = np.array(peak_memories)
    #     auc_checkpoint_name = os.path.join(
    #         self.experiment_dir, f"task{task_id}_auc.log"
    #     )
    #     torch.save(
    #         {"epochs": epochs,
    #          "success": successes,
    #          "training_loss": training_losses,
    #          "peak_memories": peak_memories,
    #          },
    #         auc_checkpoint_name,
    #     )

    def learn_one_task(self, dataset, task_id, benchmark, result_summary):

        self.start_task(task_id)

        # recover the corresponding manipulation task ids
        gsz = self.cfg.data.task_group_size
        manip_task_ids = list(range(task_id * gsz, (task_id + 1) * gsz))

        model_checkpoint_name = os.path.join(
            self.experiment_dir, f"task{task_id}_model.pth"
        )

        train_dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            sampler=RandomSampler(dataset),
            persistent_workers=True,
        )

        successes = []
        training_losses = []
        epochs = []
        peak_memories = []

        # for evaluate how fast the agent learns on current task, this corresponds
        # to the area under success rate curve on the new task.
        cumulated_counter = 0.0
        idx_at_best_succ = 0

        prev_success_rate = -1.0
        prev_train_loss = 1e5
        # best_state_dict = self.policy.state_dict()  # currently save the best model

        task = benchmark.get_task(task_id)
        task_emb = benchmark.get_task_emb(task_id)

        for _ in range(task_id):
            self.policy.add_new_and_freeze_previous(self.cfg.policy.ll_expert_per_task)

        # TODO: used to freeze prior task and train current task
        if task_id > 0:
            self.policy.load_state_dict(torch_load_model(f"/home/kavin/Documents/GitProjects/CL/DMPEL/experiments/lifelong/acil/{benchmark.name}/run_00{task_id-1}/seed_100/task{task_id-1}_model.pth")[0])
            print(f'[info] load model from experiments/lifelong/acil/{benchmark.name}/run_00{task_id-1}/seed_100/task{task_id-1}_model.pth')

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


        for epoch in range(1, self.cfg.train.n_epochs + 1):

            # # TODO: used to train analytic only
            # self.policy.load_state_dict(torch_load_model(f"/home/kavin/Documents/GitProjects/CL/DMPEL/experiments/lifelong/acil/{benchmark.name}/run_00{task_id}/seed_100/task{task_id}_model.pth")[0])
            # print(f'[info] load model from experiments/lifelong/acil/{benchmark.name}/run_00{task_id}/seed_100/task{task_id}_model.pth')
            # break

            t0 = time.time()

            self.policy.train()
            training_loss_avg = 0.0
            for (idx, data) in enumerate(train_dataloader):
                loss = self.observe(data)
                training_loss_avg += loss
                if self.scheduler is not None:
                    self.scheduler.step()

            t1 = time.time()

            if torch.cuda.is_available():
                MB = 1024.0 * 1024.0
                peak_memory = (torch.cuda.max_memory_allocated() / MB) / 1000
            else:
                peak_memory = 0

            # training_loss_avg /= len(train_dataloader)

            # TODO: update EMA
            self.policy.policy_head.ema_step()

            print(
                f'[info] # Batch: {len(train_dataloader)} | Epoch: {epoch:3d} | '
                f'train loss: {training_loss_avg:5.2f} | '
                f'time: {(t1 - t0) / 60:4.2f} | Memory utilization: {peak_memory:.2f} GB'
            )

            # if (not self.cfg.use_ddp) or (int(os.environ["RANK"]) == 0):
            #     self.summary_writer.add_scalar("bc/train_loss", training_loss_avg, epoch)
            #     self.summary_writer.add_scalar("bc/peak_memory", peak_memory, epoch)

            if epoch % self.cfg.eval.eval_every == 0:  # evaluate BC loss
                training_losses.append(training_loss_avg)
                # if training_loss_avg < prev_train_loss:
                # model_checkpoint_name = os.path.join(
                #     self.experiment_dir, f"task{task_id}_model_ep{epoch}.pth"
                # )
                torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg, learnable_only=False)
                prev_success_rate = training_loss_avg

                # t0 = time.time()
                #
                # task_str = f"k{task_id}_e{epoch//self.cfg.eval.eval_every}"
                # sim_states = (
                #     result_summary[task_str] if self.cfg.eval.save_sim_states else None
                # )
                #
                # success_rate = evaluate_one_task_success(
                #     cfg=self.cfg,
                #     algo=self,
                #     task=task,
                #     task_emb=task_emb,
                #     task_id=task_id,
                #     sim_states=sim_states,
                #     task_str="",
                # )
                #
                # epochs.append(epoch)
                # peak_memories.append(peak_memory)
                # training_losses.append(training_loss_avg)
                # successes.append(success_rate)
                #
                # self.summary_writer.add_scalar("success_rate", success_rate, epoch)
                #
                # if prev_success_rate < success_rate:
                #     torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg, learnable_only=True)
                #     prev_success_rate = success_rate
                #     idx_at_best_succ = len(training_losses) - 1
                #
                #     cumulated_counter += 1.0
                #     ci = confidence_interval(success_rate, self.cfg.eval.n_eval)
                #     tmp_successes = np.array(successes)
                #     tmp_successes[idx_at_best_succ:] = successes[idx_at_best_succ]
                #
                # t1 = time.time()
                # print(
                #     f"[info] Epoch: {epoch:3d} | succ: {success_rate:4.2f} ± {ci:4.2f} | best succ: {prev_success_rate} "
                #     + f"| succ. AoC {tmp_successes.sum()/cumulated_counter:4.2f} | time: {(t1-t0)/60:4.2f}",
                #     flush=True,
                # )

        # self.policy.load_state_dict(torch_load_model(model_checkpoint_name)[0], strict=True)
        # torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg, learnable_only=False)

        self.end_task(dataset, task_id, benchmark)

        # torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg, learnable_only=True)

        # if (not self.cfg.use_ddp or int(os.environ["RANK"]) == 0) and (not self.cfg.debug_no_eval):
        # return the metrics regarding forward transfer
        training_losses = np.array(training_losses)
        successes = np.array(successes)
        epochs = np.array(epochs)
        peak_memories = np.array(peak_memories)
        auc_checkpoint_name = os.path.join(
            self.experiment_dir, f"task{task_id}_auc.log"
        )
        torch.save(
            {"epochs": epochs,
             "success": successes,
             "training_loss": training_losses,
             "peak_memories": peak_memories,
             },
            auc_checkpoint_name,
        )

        # pretend that the agent stops learning once it reaches the peak performance
        # training_losses[idx_at_best_succ:] = training_losses[idx_at_best_succ]
        # successes[idx_at_best_succ:] = successes[idx_at_best_succ]
        # return successes.sum() / cumulated_counter, training_losses.sum() / cumulated_counter
        # else:
        #     return 0.0, 0.0
