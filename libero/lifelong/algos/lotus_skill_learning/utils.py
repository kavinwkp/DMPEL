import h5py
import glob
import re
import os
import torch
import numpy as np
from tqdm import tqdm
from libero.lifelong.algos.lotus_skill_learning.models.model_utils import safe_cuda

def get_subtask_label(idx, saved_ep_subtasks_seq, horizon):
    for (start_idx, end_idx, subtask_label) in saved_ep_subtasks_seq:
        if start_idx <= idx <= end_idx:
            return min(end_idx, idx + horizon), subtask_label

def save_subgoal_embedding(cfg, networks, data_file_name_list, skill_learning_cfg, skill_exp_name, task_id=None):
    if cfg.lifelong.phase == 'base':
        subgoal_embedding_file_name = os.path.join(cfg.experiment_dir, f"subgoal_embedding.hdf5")
    elif cfg.lifelong.phase == "lifelong":
        subgoal_embedding_file_name = os.path.join(cfg.experiment_dir, f"subgoal_embedding_{task_id}.hdf5")
    subgoal_embedding_file = h5py.File(subgoal_embedding_file_name, "w")
    for data_file_name in tqdm(data_file_name_list):
        dataset_category, dataset_name = data_file_name.split("/")[-2:]
        dataset_name = dataset_name.split(".")[0]
        demo_file = h5py.File(f"{data_file_name}", "r")
        if cfg.lifelong.phase == 'base':
            file_pattern = f"/public/home/group_luoping/leiyuheng/dataset/datasets/lotus/{skill_exp_name}/skill_data/{dataset_category}/{dataset_name}*"
        elif cfg.lifelong.phase == "lifelong":
            file_pattern = f"/public/home/group_luoping/leiyuheng/dataset/datasets/lotus/{skill_exp_name}/task_" + str(91+task_id) + "/skill_data" + f"/{dataset_category}/{dataset_name}*"
        matching_files = glob.glob(file_pattern)
        # print(file_pattern)
        # print(matching_files)
        assert len(matching_files) == 1
        subtasks_file_name = matching_files[0]
        subtask_file = h5py.File(subtasks_file_name, "r")
        
        if cfg.goal_modality == "dinov2":
            dinov2_feature_file_name = re.sub(r"(datasets/)([^/]+)(/)", r"\1dinov2/\2\3", data_file_name)
            dinov2_feature_file = h5py.File(dinov2_feature_file_name, "r")

        demo_num = len(demo_file['data'].keys())
        grp = subgoal_embedding_file.create_group(f"{dataset_name}")
        for ep_idx in range(demo_num):
            # Generate embedding
            if f"demo_subtasks_seq_{ep_idx}" not in subtask_file["subtasks"]:
                continue
            saved_ep_subtasks_seq = subtask_file["subtasks"][f"demo_subtasks_seq_{ep_idx}"][()]
            agentview_images = demo_file[f"data/demo_{ep_idx}/obs/agentview_rgb"][()]
            eye_in_hand_images = demo_file[f"data/demo_{ep_idx}/obs/eye_in_hand_rgb"][()]
            ee_states = demo_file[f"data/demo_{ep_idx}/obs/ee_states"][()]
            gripper_states = demo_file[f"data/demo_{ep_idx}/obs/gripper_states"][()]
            joint_states = demo_file[f"data/demo_{ep_idx}/obs/joint_states"][()]
            if cfg.goal_modality == "dinov2":
                dinov2_embedding = dinov2_feature_file[f"data/demo_{ep_idx}/embedding"][()]


            embeddings = []
            for i in range(len(agentview_images)):
                future_idx, subtask_label = get_subtask_label(i, saved_ep_subtasks_seq, horizon=skill_learning_cfg.skill_subgoal_cfg.horizon)
                agentview_image = safe_cuda(torch.from_numpy(np.array(agentview_images[future_idx]).transpose(2, 0, 1)).unsqueeze(0)).float() / 255.
                eye_in_hand_image = safe_cuda(torch.from_numpy(np.array(eye_in_hand_images[future_idx]).transpose(2, 0, 1)).unsqueeze(0)).float() / 255.
                # if cfg.goal_modality == "BUDS":
                if skill_learning_cfg.skill_subgoal_cfg.use_eye_in_hand:
                    state_image = torch.cat([agentview_image, eye_in_hand_image], dim=1)
                else:
                    state_image = agentview_image
                embedding = networks[subtask_label].get_embedding(state_image).detach().cpu().numpy().squeeze()
                # elif cfg.goal_modality == "ee_states":
                #     embedding = np.concatenate([ee_states[future_idx], gripper_states[future_idx]])
                # elif cfg.goal_modality == "joint_states":
                #     embedding = np.concatenate([joint_states[future_idx], gripper_states[future_idx]])
                # elif cfg.goal_modality == "dinov2":
                #     embedding = dinov2_embedding[future_idx]
                # import ipdb; ipdb.set_trace()
                embeddings.append(embedding)

            ep_data_grp = grp.create_group(f"demo_{ep_idx}")
            ep_data_grp.create_dataset("embedding", data=np.stack(embeddings, axis=0))
        subtask_file.close()
        demo_file.close()
        grp.attrs["embedding_dim"] = len(embeddings[-1])
    subgoal_embedding_file.close()

