#!/bin/bash

python libero/lifelong/main.py seed=100 benchmark_name=libero_goal \
        policy=bc_foundation_iscil_policy lifelong=iscil \
        exp=/public/home/group_luoping/leiyuheng/experiments/ckpt/lifelong/iscil/goal \
        pretrain_model_path=/public/home/group_luoping/leiyuheng/experiments/pretraining/chunkonly_fft/clip_base_libero90/seed_100/multitask_model_ep10.pth