#!/bin/bash

python libero/lifelong/main.py seed=100 benchmark_name=libero_goal \
        policy=bc_foundation_policy_fft lifelong=base \
        exp=/public/home/group_luoping/leiyuheng/experiments/ckpt/lifelong/seq_fft_nopt/goal