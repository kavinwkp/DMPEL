"""
This file contains modules that encode language task embeddings.
"""
from typing import Optional
import torch.nn as nn
import torch

class IdentityEncoder(nn.Module):
    """
    Dummy encoder that directly outputs the pretrained task embedding
    """

    def __init__(self, input_size, sentence_length=None, dummy=True):
        super().__init__()
        self.input_size = input_size
        self.sentence_length = sentence_length


    def forward(self, data):
        """
        data:
            task_emb: (B, E)
        """
        h = data["task_emb"]  # (B, L, H)
        return h


class MLPEncoder(nn.Module):
    """
    Encode task embedding

    h = f(e), where
        e: pretrained task embedding from large model
        h: latent embedding (B, H)
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        assert num_layers >= 1, "[error] num_layers < 1"
        sizes = [input_size] + [hidden_size] * (num_layers - 1) + [output_size]
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.projection = nn.Sequential(*layers)

    def forward(self, data):
        """
        data:
            task_emb: (B, E)
        """
        h = self.projection(data["task_emb"])  # (B, H)
        return h

from transformers import AutoTokenizer, AutoModel
from libero.lifelong.models.modules.adapter import LoRA, L2MLoRA

class ClipTextEncoder(nn.Module):
    def __init__(self, lora_cfg, use_lora: bool = False, fullft: bool = False,
                 model_name: str = "openai/clip-vit-base-patch16"):
        super().__init__()
        # self.tz = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        model = AutoModel.from_pretrained(model_name)
        # model name could be:
        # "openai/clip-vit-base-patch16"
        # "openai/clip-vit-large-patch14"
        # "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        self.model = model.base_model.text_model
        self.post_proj = model.base_model.text_projection
        self.use_lora = use_lora
        self.fullft = fullft
        for param in self.model.parameters():
            param.requires_grad = fullft
        for param in self.post_proj.parameters():
            param.requires_grad = fullft
        if self.use_lora == "None":
            pass
        else:
            self.lora_rank = lora_cfg.rank
            if lora_cfg.lora_layers_list == "all":
                self.lora_layers_list = list(range(len(self.model.encoder.layers)))
            else:
                self.lora_layers_list = lora_cfg.lora_layers_list
            assert isinstance(self.lora_layers_list, list)
            for i, block in enumerate(self.model.encoder.layers):
                if i in self.lora_layers_list:
                    orig_q = block.self_attn.q_proj
                    orig_v = block.self_attn.v_proj
                    dim = orig_q.in_features
                    if self.use_lora == "L2MLoRA":
                        self.pool_size = lora_cfg.pool_size
                        lora_q = L2MLoRA(orig_q, self.pool_size, dim, self.lora_rank)
                        lora_v = L2MLoRA(orig_v, self.pool_size, dim, self.lora_rank)
                    elif self.use_lora == "LoRA":
                        lora_q = LoRA(orig_q, dim, self.lora_rank)
                        lora_v = LoRA(orig_v, dim, self.lora_rank)
                    else:
                        raise NotImplementedError
                    setattr(block.self_attn, 'q_proj', lora_q)
                    setattr(block.self_attn, 'v_proj', lora_v)
    
    def forward(self, tokens):
        out = self.model(**tokens)
        out = self.post_proj(out["pooler_output"])
        return out
