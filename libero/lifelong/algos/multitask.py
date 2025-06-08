import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from libero.lifelong.algos.base import Sequential
from libero.lifelong.metric import *
from libero.lifelong.models import *
from libero.lifelong.utils import *


class Multitask(Sequential):
    """
    The multitask learning baseline/upperbound.
    """

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)

    def learn_all_tasks(self, datasets, benchmark, result_summary):
        self.start_task(-1)
        concat_dataset = ConcatDataset(datasets)

        # learn on all tasks, only used in multitask learning
        model_checkpoint_name = os.path.join(
            self.experiment_dir, f"multitask_model.pth"
        )
        all_tasks = list(range(benchmark.n_tasks))

        multiprocessing.set_start_method("fork", force=True)
        if self.cfg.use_ddp:
            torch.distributed.barrier()
            train_dataloader = DataLoader(
                    concat_dataset,
                    batch_size=self.cfg.train.batch_size,
                    num_workers=self.cfg.train.num_workers,
                    shuffle=False,
                    sampler=DistributedSampler(concat_dataset),
                    persistent_workers=True,
            )
        else:
            train_dataloader = DataLoader(
                    concat_dataset,
                    batch_size=self.cfg.train.batch_size,
                    num_workers=self.cfg.train.num_workers,
                    sampler=RandomSampler(concat_dataset),
                    persistent_workers=True,
                )

        if self.cfg.train.scheduler is not None:
            self.cfg.train.scheduler.kwargs["epochs"] = self.cfg.train.n_epochs
            self.scheduler = eval(self.cfg.train.scheduler.name)(
                self.optimizer,
                steps_per_epoch = len(train_dataloader),
                # T_max=self.cfg.train.n_epochs
                **self.cfg.train.scheduler.kwargs,
            )
        
        if not self.cfg.use_ddp or int(os.environ["RANK"]) == 0:
            successes = []
            losses = []
            epochs = []
            peak_memories = []
            times = []
            
            # for evaluate how fast the agent learns on current task, this corresponds
            # to the area under success rate curve on the new task.
            cumulated_counter = 0.0
            idx_at_best_succ = 0

            prev_success_rate = -1.0

        # start training
        for epoch in range(0, self.cfg.train.n_epochs + 1):

            t0 = time.time()

            if self.cfg.use_ddp:
                train_dataloader.sampler.set_epoch(epoch)            

            if epoch > 0 or (self.cfg.pretrain):  # update
                self.policy.train()
                training_loss = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    loss = self.observe(data)
                    training_loss += loss
                    if self.scheduler is not None:
                        self.scheduler.step()
                # training_loss /= len(train_dataloader)
            else:  # just evaluate the zero-shot performance on 0-th epoch
                training_loss = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    loss = self.eval_observe(data)
                    training_loss += loss
                # training_loss /= len(train_dataloader)
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
                        f'[info] GPU: {int(os.environ["RANK"])} | # Batch: {sum(dataloader_len_gather_list).item()} | Epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1-t0)/60:4.2f} | '
                        f'Memory utilization: %.3f GB' % peak_memory,
                        flush=True,
                    )

            else:
                training_loss /= len(train_dataloader)
                print(
                    f'[info] # Batch: {len(train_dataloader)} | Epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1-t0)/60:4.2f} | '
                    f'Memory utilization: %.3f GB' % peak_memory,
                    flush=True,
                )

            if (not self.cfg.use_ddp) or (int(os.environ["RANK"]) == 0):
                self.summary_writer.add_scalar("train_loss", training_loss, epoch)
                self.summary_writer.add_scalar("memory_utilization", peak_memory, epoch)
                self.summary_writer.add_scalar("time", (t1-t0)/60, epoch)
                # for name, param in self.policy.named_parameters():
                    # if name.startswith("pearl"):
                        # self.summary_writer.add_histogram(f'value/{name}', param, epoch)
                        # if param.grad is not None:
                        #     self.summary_writer.add_histogram(f'grad/{name}.grad', param.grad, epoch)
            
            if epoch % self.cfg.eval.eval_every == 0 and epoch > 0 and (not self.cfg.use_ddp or int(os.environ["RANK"]) == 0):  # evaluate BC loss
                t0 = time.time()
                
                self.policy.eval()

                model_checkpoint_name_ep = os.path.join(
                    self.experiment_dir, f"multitask_model_ep{epoch}.pth"
                )
                torch_save_model(self.policy, model_checkpoint_name_ep, cfg=self.cfg, learnable_only=True)
                losses.append(training_loss)

                # for multitask learning, we provide an option whether to evaluate
                # the agent once every eval_every epochs on all tasks, note that
                # this can be quite computationally expensive. Nevertheless, we
                # save the checkpoints, so users can always evaluate afterwards.
                if self.cfg.lifelong.eval_in_train:
                    success_rates = evaluate_multitask_training_success(
                        self.cfg, self, benchmark, all_tasks
                    )
                    success_rate = np.mean(success_rates)
                else:
                    success_rate = 0.0
                successes.append(success_rate)
                epochs.append(epoch)
                peak_memories.append(peak_memory)

                # self.summary_writer.add_scalar("success_rate", success_rate, epoch)

                # if prev_success_rate < success_rate and (not self.cfg.pretrain):
                #     # torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg, learnable_only=True)
                #     prev_success_rate = success_rate
                #     idx_at_best_succ = len(losses) - 1

                t1 = time.time()
                cumulated_counter += 1.0
                ci = confidence_interval(success_rate, self.cfg.eval.n_eval)
                tmp_successes = np.array(successes)
                tmp_successes[idx_at_best_succ:] = successes[idx_at_best_succ]

                if self.cfg.lifelong.eval_in_train:
                    print(
                        f"[info] Epoch: {epoch:3d} | succ: {success_rate:4.2f} ± {ci:4.2f} | best succ: {prev_success_rate} "
                        + f"| succ. AoC {tmp_successes.sum()/cumulated_counter:4.2f} | time: {(t1-t0)/60:4.2f}",
                        flush=True,
                    )

        # load the best policy if there is any
        if self.cfg.lifelong.eval_in_train:
            self.policy.load_state_dict(torch_load_model(model_checkpoint_name)[0], strict=False)
        
        self.end_task(concat_dataset, -1, benchmark)

        if not self.cfg.use_ddp or int(os.environ["RANK"]) == 0:    
            # return the metrics regarding forward transfer
            losses = np.array(losses)
            successes = np.array(successes)
            epochs = np.array(epochs)
            peak_memories = np.array(peak_memories)
            auc_checkpoint_name = os.path.join(
                self.experiment_dir, f"multitask_auc.log"
            )
            torch.save(
                {   "epochs": epochs,
                    "success": successes,
                    "loss": losses,
                    "peak_memories": peak_memories,
                    "times": times,
                },
                auc_checkpoint_name,
            )

            if self.cfg.lifelong.eval_in_train:
                loss_at_best_succ = losses[idx_at_best_succ]
                success_at_best_succ = successes[idx_at_best_succ]
                losses[idx_at_best_succ:] = loss_at_best_succ
                successes[idx_at_best_succ:] = success_at_best_succ
            return successes.sum() / cumulated_counter, losses.sum() / cumulated_counter
        else:
            return 0.0, 0.0
