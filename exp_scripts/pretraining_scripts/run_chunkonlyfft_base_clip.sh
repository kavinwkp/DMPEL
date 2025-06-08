#!/bin/bash

#SBATCH -J chunkonly_fft_clip_base_liberogoal
#SBATCH -o /mnt/hwfile/gveval/leiyuheng/libero/experiments/log/pretraining/chunkonly_fft/clip_base_libero90_seed_100_log.out
#SBATCH -e /mnt/hwfile/gveval/leiyuheng/libero/experiments/log/pretraining/chunkonly_fft/clip_base_libero90_seed_100_err.err

torchrun --standalone --nproc_per_node=gpu libero/lifelong/main.py \
            seed=100 benchmark_name=libero_90 policy=bc_foundation_policy_fft \
            lifelong=multitask exp=/mnt/hwfile/gveval/leiyuheng/libero/experiments/ckpt/pretraining/chunkonly_fft/clip_base_libero90 \
            task_embedding_format=clip text_tokens_or_embeddings=tokens use_ddp=true \
            train.n_epochs=10 debug_no_eval=false train.batch_size=32
