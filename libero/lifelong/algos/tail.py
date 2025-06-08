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

class TAIL(Sequential):
    """
    TAIL method
    """

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)

    def start_task(self, task):
        super().start_task(task)
        if self.current_task > 0:
            model_checkpoint_name = os.path.join(self.experiment_dir, f"task{task-1}_model.pth")
            state_dict = torch_load_model(model_checkpoint_name)[0]
            for k, v in state_dict.items():
                noise = torch.randn_like(v.float(), dtype=torch.float) * 1e-3
                state_dict[k] = v + noise
            msg = self.policy.load_state_dict(state_dict, strict=False)