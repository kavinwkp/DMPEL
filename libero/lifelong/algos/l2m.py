import collections
import time
import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import ConcatDataset, RandomSampler

from libero.lifelong.algos.base import Sequential
from libero.lifelong.datasets import TruncatedSequenceDataset
from libero.lifelong.utils import *
from libero.lifelong.metric import *
from libero.lifelong.models import *


from torch.utils.tensorboard import SummaryWriter

class L2M(Sequential):
    """
    TAIL method
    """

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        setattr(self.policy, "n_tasks", n_tasks)
        setattr(self.policy, "oracle", cfg.lifelong.oracle)
        
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
        if self.cfg.lifelong.oracle:
            self.policy.set_task_id(task)