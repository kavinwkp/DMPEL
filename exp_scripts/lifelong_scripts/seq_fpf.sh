#!/bin/bash

python libero/lifelong/main.py seed=100 benchmark_name=libero_goal \
        policy=bc_foundation_policy_frozen lifelong=base \
        exp=/public/home/group_luoping/leiyuheng/experiments/ckpt/lifelong/seq_fpf/goal \
        pretrain_model_path=/public/home/group_luoping/leiyuheng/experiments/pretraining/chunkonly_frozen/clip_base_libero90/seed_100/multitask_model_ep10.pth