#!/bin/bash

#SBATCH --job-name=ewc_goal_100

torchrun --standalone --nproc_per_node=gpu libero/lifelong/main.py seed=100 benchmark_name=libero_goal \
        policy=bc_foundation_policy_fft lifelong=ewc \
        exp=/public/home/group_luoping/leiyuheng/experiments/ckpt/lifelong/ewc/goal \
        use_ddp=true train.batch_size=16 \
        pretrain_model_path=/public/home/group_luoping/leiyuheng/experiments/pretraining/chunkonly_fft/clip_base_libero90/seed_100/multitask_model_ep10.pth