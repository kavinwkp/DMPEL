import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
from hydra.utils import to_absolute_path
from thop import profile
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, CLIPTextModel, logging


def control_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def safe_device(x, device="cpu"):
    if device == "cpu":
        return x.cpu()
    elif "cuda" in device:
        if torch.cuda.is_available():
            return x.to(device)
        else:
            return x.cpu()


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def torch_save_model(model, model_path, cfg=None, previous_masks=None, learnable_only=False):
    state_dict = model.state_dict()
    if learnable_only:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                del state_dict[name]
    torch.save(
        {
            "state_dict": state_dict,
            "cfg": cfg,
            "previous_masks": previous_masks,
        },
        model_path,
    )

def torch_load_model(model_path, map_location=None):
    model_dict = torch.load(model_path, map_location=map_location)
    cfg = None
    if "cfg" in model_dict:
        cfg = model_dict["cfg"]
    if "previous_masks" in model_dict:
        previous_masks = model_dict["previous_masks"]
    return model_dict["state_dict"], cfg, previous_masks


def get_train_test_loader(
    dataset, train_ratio, train_batch_size, test_batch_size, num_workers=(0, 0)
):

    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=num_workers[0],
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        num_workers=num_workers[1],
        shuffle=False,
    )
    return train_dataloader, test_dataloader


def confidence_interval(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n)


def count_policy_parameters(algo, cfg):
    model = copy.deepcopy(algo.policy)
    # tmp_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)
    # data = next(iter(tmp_loader))
    # data = TensorUtils.map_tensor(data, lambda x: safe_device(x, device=cfg.device))
    # tensor_list = list(algo.policy.state_dict().items())
    counter_list = {"image_encoder_spatial": 0,
                    "language_encoder_spatial": 0,
                    "extra_encoder": 0,
                    "fusion_module": 0,
                    "temporal_transformer": 0,
                    "policy_head": 0,
                    # "spatial_lora": 0,
                    # "temporal_lora": 0,
                    "pearl": 0,}
    for layer_tensor_name, tensor in model.named_parameters():
        for key in counter_list.keys():
            if layer_tensor_name.startswith(key) and tensor.requires_grad:
                counter_list[key] += torch.numel(tensor)
        if "moe_router" in layer_tensor_name and tensor.requires_grad:
            counter_list["pearl"] += torch.numel(tensor)
        # if "A" in layer_tensor_name or "B" in layer_tensor_name:
        #     if "spatial" in layer_tensor_name:
        #         counter_list["spatial_lora"] += torch.numel(tensor)
        #     elif "temporal" in layer_tensor_name:
        #         counter_list["temporal_lora"] += torch.numel(tensor)
        print('{}: {}, {}'.format(layer_tensor_name, torch.numel(tensor), tensor.requires_grad))
    print(counter_list)
    # macs, params = profile(model, inputs=(data,), verbose=False, report_missing=True)
    # from torchtnt.utils.flops import FlopTensorDispatchMode
    # with FlopTensorDispatchMode(model) as ftdm:
    #     _ = model(data)
    #     flops = copy.deepcopy(ftdm.flop_counts)
    #     sum_flops = sum(flops[''].values())
    # GFLOPs = sum_flops * 2 / 1e9
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: {}".format(total_params))
    trainable_params = sum(p.numel() for p in model.parameters() if (not p.requires_grad))
    print("Frozen parameters: {}".format(trainable_params))
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters: {}".format(trainable_params))
    del model
    return counter_list, total_params, trainable_params
    # return GFLOPs, MParams


def create_experiment_dir(cfg):
    # prefix = "experiments_foundation_base"
    # prefix = "experiments"
    # if cfg.pretrain_model_path != "":
    #     prefix += "_finetune"
    # if cfg.data.task_order_index > 0:
    #     prefix += f"_permute{cfg.data.task_order_index}"
    # if cfg.task_embedding_format == "one-hot":
    #     prefix += f"_onehot"
    # if cfg.task_embedding_format == "clip":
    #     prefix += f"_clip"
    # if cfg.task_embedding_format == "bert":
    #     prefix += f"_bert"

    # experiment_dir = (
    #     f"./{prefix}/{cfg.exp}/seed_{cfg.seed}"
    # )
    experiment_dir = (
        f"{cfg.exp}/seed_{cfg.seed}"
    )

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    if cfg.debug_no_eval:
        # look for the most recent run
        experiment_id = 0
        for path in Path(experiment_dir).glob("run_*"):
            if not path.is_dir():
                continue
            try:
                folder_id = int(str(path).split("run_")[-1])
                if folder_id > experiment_id:
                    experiment_id = folder_id
            except BaseException:
                pass
        experiment_id += 1
        experiment_dir += f"/run_{experiment_id:03d}"
    cfg.experiment_dir = experiment_dir
    cfg.experiment_name = "_".join(cfg.experiment_dir.split("/")[2:])
    os.makedirs(cfg.experiment_dir, exist_ok=True)
    return True


def get_task_embs(cfg, descriptions):
    logging.set_verbosity_error()
    if cfg.task_embedding_format == "clip":
        model_name = cfg.policy.language_encoder.network_kwargs.model_name
        tz = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=cfg.data.max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        tokens = {"attention_mask": tokens["attention_mask"], "input_ids": tokens["input_ids"]}
        model_noproj = model.base_model.text_model
        text_proj = model.base_model.text_projection
        out = model_noproj(**tokens)
        last_hidden_states = text_proj(out["last_hidden_state"]).detach()
        task_embs = model.get_text_features(**tokens).detach()
    else:
        raise NotImplementedError
    if cfg.text_tokens_or_embeddings == "tokens":
        tokens_task_by_task = {}
        for i in range(len(descriptions)):
            tokens_task_by_task[i] = {"attention_mask": tokens["attention_mask"][i], "input_ids": tokens["input_ids"][i]}
        return tokens_task_by_task
    elif cfg.text_tokens_or_embeddings == "embeddings":
        cfg.policy.language_encoder.network_kwargs.input_size = task_embs.shape[-1]
        return task_embs
    elif cfg.text_tokens_or_embeddings == "hidden_states":
        cfg.policy.language_encoder.network_kwargs.sentence_length = last_hidden_states.shape[-2]
        cfg.policy.language_encoder.network_kwargs.input_size = task_embs.shape[-1]
        last_hidden_states = torch.cat([task_embs.unsqueeze(dim=1), last_hidden_states], dim=1)
        return last_hidden_states
    else:
        raise NotImplementedError