import argparse
import sys
import os

# TODO: find a better way for this?
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import hydra
import json
import numpy as np
import pprint
import time
import torch
import wandb
import yaml
from easydict import EasyDict
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoModel, pipeline, AutoTokenizer, logging
from pathlib import Path

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter
from libero.lifelong.algos import *
from libero.lifelong.datasets import get_dataset, SequenceVLDataset, GroupedTaskDataset
from libero.lifelong.metric import (
    evaluate_loss,
    evaluate_success,
    raw_obs_to_tensor_obs,
)
from libero.lifelong.utils import (
    control_seed,
    safe_device,
    torch_load_model,
    NpEncoder,
    get_task_embs,
    # compute_flops,
)


import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils

import time


benchmark_map = {
    "libero_10": "LIBERO_10",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
    "libero_100": "LIBERO_100",
}

algo_map = {
    "base": "Sequential",
    "er": "ER",
    "ewc": "EWC",
    "packnet": "PackNet",
    "multitask": "Multitask",
    "tail": "TAIL",
    "dmpel": "DMPEL",
    "acil": "ACILLearner"
}

policy_map = {
    "bc_rnn_policy": "BCRNNPolicy",
    "bc_transformer_policy": "BCTransformerPolicy",
    "bc_vilt_policy": "BCViLTPolicy",
    "bc_foundation_tail_policy": "BCFoundationTailPolicy",
    "bc_foundation_dmpel_policy": "BCFoundationDmpelPolicy",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--experiment_dir", type=str, default="experiments")
    # for which task suite
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        # choices=["libero_10", "libero_spatial", "libero_object", "libero_goal", "libero_100"],
    )
    parser.add_argument("--task_id", type=int, required=False)
    # method detail
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        # choices=["base", "er", "ewc", "packnet", "multitask"],
    )
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        # choices=["bc_rnn_policy", "bc_transformer_policy", "bc_vilt_policy"],
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", type=int, required=True)
    parser.add_argument("--ep", type=int, help="epoch number of which .pth", default=0)
    # parser.add_argument("--load_task", type=int, help="for single task")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--save-videos", action="store_true")
    # parser.add_argument('--save_dir',  type=str, required=True)
    args = parser.parse_args()
    args.device_id = "cuda:" + str(args.device_id)
    args.save_dir = f"{args.experiment_dir}_saved"

    # if args.algo == "multitask":
    #     assert args.ep in list(
    #         range(0, 55, 5)
    #     ), "[error] ep should be in [0, 5, ..., 50]"
    # else:
    #     assert args.load_task in list(
    #         range(10)
    #     ), "[error] load_task should be in [0, ..., 9]"
    return args


def main(args):
    control_seed(args.seed)


    model = args.model

    run_folder = args.experiment_dir
    try:
        if args.algo == "multitask":
            model_path = os.path.join(run_folder, f"multitask_model_ep{args.ep}.pth")
            sd, cfg, previous_mask = torch_load_model(model_path, map_location=args.device_id)
        else:
            if args.ep == 0:
                model_path = os.path.join(run_folder, f"task{model}_model.pth")
            else:
                model_path = os.path.join(run_folder, f"task{model}_model_ep{args.ep}.pth")
            sd, cfg, previous_mask = torch_load_model(model_path, map_location=args.device_id)
    except:
        print(f"[error] cannot find the checkpoint at {str(model_path)}")
        sys.exit(0)

    cfg.folder = get_libero_path("datasets")
    cfg.bddl_folder = get_libero_path("bddl_files")
    cfg.init_states_folder = get_libero_path("init_states")

    cfg.device = args.device_id
    algo = safe_device(eval(algo_map[args.algo])(10, cfg), cfg.device)
    # print(algo.policy)
    # algo.policy.previous_mask = previous_mask

    # if cfg.lifelong.algo == "PackNet":
    #     algo.eval()
    #     for module_idx, module in enumerate(algo.policy.modules()):
    #         if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
    #             weight = module.weight.data
    #             mask = algo.previous_masks[module_idx].to(cfg.device)
    #             weight[mask.eq(0)] = 0.0
    #             weight[mask.gt(args.task_id + 1)] = 0.0
    #             # we never train norm layers
    #         if "BatchNorm" in str(type(module)) or "LayerNorm" in str(type(module)):
    #             module.eval()

    # algo.policy.load_state_dict(sd)
    if cfg.lifelong.algo == "DMPEL" or cfg.lifelong.algo == "ACILLearner":
        algo.policy.init_moe_policy()
        # print(algo.policy)
        for i in range(model + 1):
            algo.policy.add_new_and_freeze_previous(add_expert_num=1)
        # algo = safe_device(algo, cfg.device)
    elif cfg.lifelong.algo == "TAIL":
        algo.policy.init_lora()

    algo = safe_device(algo, cfg.device)
    algo.policy.load_state_dict(sd, strict=True)

    # if cfg.lifelong.algo == "TAIL":   # acil
    # algo.policy.init_policy_head()
    # policy_head_path = os.path.join(run_folder, f"policy{model}_head.pth")
    # sd_policy_head, cfg, previous_mask = torch_load_model(policy_head_path, map_location=args.device_id)
    # algo.policy.policy_head.load_state_dict(sd_policy_head)

    if cfg.lifelong.algo == "ACILLearner":
        algo.policy.init_router()
        # router_path = os.path.join(run_folder, f"acil{args.task_id}_router.pth")
        router_path = os.path.join(run_folder, f"acil{model}_router.pth")
        sd, cfg, previous_mask = torch_load_model(router_path, map_location=args.device_id)
        algo.policy.acil_router.load_state_dict(sd)

    if not hasattr(cfg.data, "task_order_index"):
        cfg.data.task_order_index = 0

    # get the benchmark the task belongs to
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    descriptions = [benchmark.get_task(i).language for i in range(benchmark.n_tasks)]

    # task_embs = get_task_embs(cfg, descriptions)
    # task_embs_dir = os.path.join('/home/kavin/Documents/GitProjects/CL/DMPEL/clip', benchmark.name)
    # os.makedirs(task_embs_dir, exist_ok=True)
    # task_embs_file = os.path.join(task_embs_dir, 'task_embs.pt')
    #
    # if os.path.exists(task_embs_file):
    #     print(f"[info] Loading task embeddings from {task_embs_file}")
    #     task_embs = torch.load(task_embs_file)
    # else:
    task_embs = get_task_embs(cfg, descriptions)  # (n_tasks, emb_dim)
    # torch.save(task_embs, task_embs_file)
    benchmark.set_task_embs(task_embs)

    task = benchmark.get_task(args.task_id)

    ### ======================= start evaluation ============================

    # 1. evaluate dataset loss
    try:
        dataset, shape_meta = get_dataset(
            dataset_path=os.path.join(
                cfg.folder, benchmark.get_task_demonstration(args.task_id)
            ),
            obs_modality=cfg.data.obs.modality,
            initialize_obs_utils=True,
            seq_len=cfg.data.seq_len,
        )
        # dataset = GroupedTaskDataset(
        #     [dataset], task_embs[args.task_id : args.task_id + 1]
        # )
    except Exception as e:
        print(
            f"[error] failed to load task {args.task_id} name {benchmark.get_task_names()[args.task_id]}"
        )
        print(f"[error] {e}")
        sys.exit(0)

    algo.eval()

    test_loss = 0.0

    # 2. evaluate success rate
    video_folder = os.path.join(
        args.save_dir,
        f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_on_task{args.task_id}_videos",
    )

    if args.algo == "multitask":
        save_folder = os.path.join(
            video_folder,
            f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_ep{args.ep}_on{args.task_id}.stats",
        )
    else:
        save_folder = os.path.join(
            video_folder,
            f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_on{args.task_id}.stats",
        )



    with Timer() as t, VideoWriter(video_folder, args.save_videos) as video_writer:
        env_args = {
            "bddl_file_name": os.path.join(
                cfg.bddl_folder, task.problem_folder, task.bddl_file
            ),
            "camera_heights": cfg.data.img_h,
            "camera_widths": cfg.data.img_w,
        }

        env_num = 20
        env = SubprocVectorEnv(
            [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
        )
        env.reset()
        env.seed(cfg.seed)
        algo.reset()

        init_states_path = os.path.join(
            cfg.init_states_folder, task.problem_folder, task.init_states_file
        )
        init_states = torch.load(init_states_path)
        # indices = np.arange(env_num) % init_states.shape[0]
        indices = np.arange(10, env_num+10) % init_states.shape[0]
        init_states_ = init_states[indices]

        dones = [False] * env_num
        steps = 0
        obs = env.set_init_state(init_states_)
        task_emb = benchmark.get_task_emb(args.task_id)

        num_success = 0
        for _ in range(5):  # simulate the physics without any actions
            env.step(np.zeros((env_num, 7)))

        with torch.no_grad():
            while steps < cfg.eval.max_steps:
                steps += 1

                data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                actions = algo.policy.get_action(data)
                obs, reward, done, info = env.step(actions)
                video_writer.append_vector_obs(
                    obs, dones, camera_name="agentview_image"
                )

                # check whether succeed
                for k in range(env_num):
                    dones[k] = dones[k] or done[k]
                if all(dones):
                    break

            for k in range(env_num):
                num_success += int(dones[k])

        success_rate = num_success / env_num
        env.close()

        eval_stats = {
            "loss": test_loss,
            "success_rate": success_rate,
        }

        # os.system(f"mkdir -p {args.save_dir}")
        # torch.save(eval_stats, save_folder)

        os.makedirs(video_folder, exist_ok=True)
        with open(save_folder, "w") as f:
            json.dump(eval_stats, f, cls=NpEncoder, indent=4)

    # print(algo.policy.expert_count)

    # combined = torch.cat(algo.policy.expert_count, dim=0)
    # print(combined)
    # # 统计每一列中0、1、2、3的出现次数
    # counts = torch.zeros((10, combined.shape[1]), dtype=torch.int64)  # (10, 6)
    #
    # for j in range(combined.shape[1]):
    #     col = combined[:, j]
    #     counts[:, j] = torch.bincount(col, minlength=10)
    #
    # print("\n每一列中0、1、2、3的出现次数：")
    # print(counts)
    #
    # num_splits = combined.shape[1] // 6
    # # print(num_splits)
    # new_array = torch.zeros((counts.shape[0], 6), dtype=torch.float32)
    #
    # for i in range(6):
    #     start_col = i * num_splits
    #     end_col = start_col + num_splits
    #     summed = torch.sum(counts[:, start_col:end_col], dim=1)
    #     new_array[:, i] = summed
    # # print(new_array)
    #
    # total_elements_per_column = combined.shape[0]
    #
    # frequencies = new_array / total_elements_per_column
    # print(frequencies)

    print(f"[info] finish for ckpt at {run_folder} in {t.get_elapsed_time()} sec for rollouts")
    print(f"Results are saved at {save_folder}")
    print(test_loss, success_rate)
    return success_rate


if __name__ == "__main__":
    # main()
    args = parse_args()
    success_ep = []
    # task_id = args.task_id + 1
    # for i in range(task_id):
    #     args.task_id = i
    success_ep.append(main(args))
    avg_success_ep = sum(success_ep) / len(success_ep)
    print(f"[info] success_rate: {success_ep}")
    print(f"[info] average success_rate: {avg_success_ep}")
    print(f"############## {args.task_id} ############## ")
    print()