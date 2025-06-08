import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
import warnings
from typing import Any, Union, List
from pkg_resources import packaging
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from einops import rearrange, repeat
from torch import amp

from libero.lifelong.models.modules.rgb_modules import *
from libero.lifelong.models.modules.language_modules import *
from libero.lifelong.models.modules.transformer_modules import *
from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.bc_foundation_tail_policy import BCFoundationTailPolicy
from libero.lifelong.models.policy_head import *
from libero.lifelong.models.bc_transformer_policy import ExtraModalityTokens

from libero.lifelong.models.modules.adapter import *


def reshape_transform(tensor, h, w):
    B, _, E = tensor.shape
    result = tensor[:, 1 : 1 + h * w, :].reshape(B, h, w, E)
    return result.permute(0, 3, 1, 2)

def _convert_image_to_rgb(image):
    return image.convert("RGB")


class BCFoundationISCILPolicy(BCFoundationTailPolicy):
    """
    Input: (o_{t-H}, ... , o_t)
    Output: a_t or distribution of a_t
    """

    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        self.pool_size = self.cfg.policy.adapter.pool_size # num_tasks
        self.base_per_proto = cfg.policy.base_per_proto
        self.image_embed_dim = self.image_encoder_spatial.model.embed_dim
        self.language_embed_dim = self.language_encoder_spatial.model.config.hidden_size
        self.multifacet_prototype = nn.Parameter(torch.zeros((self.pool_size, self.base_per_proto, self.image_embed_dim*2+self.language_embed_dim)), requires_grad=False)
        self.n_tasks = self.pool_size # set in L2M algo
        self.task_id = None
        self.task_to_mask = None
        self.pos_sim = True
        # set all parameters (except modulators and keys) to not require gradients
        for n, p in self.named_parameters():
            if ('A' in n or 'B' in n):
                p.requires_grad = True
            else:
                p.requires_grad = False
        self.lora_only = cfg.policy.lora_only
        if not self.lora_only:
            for i, encoder in enumerate(self.extra_encoder.encoders):
                for j, orig_linear in enumerate(encoder):
                    if isinstance(orig_linear, nn.Linear):
                        iscil_linear = L2MLinear(orig_linear, self.pool_size)
                        self.extra_encoder.encoders[i][j] = iscil_linear
            for i, orig_linear in enumerate(self.fusion_module):
                if isinstance(orig_linear, nn.Linear):
                    iscil_linear = L2MLinear(orig_linear, self.pool_size)
                    self.fusion_module[i] = iscil_linear
            for i, orig_linear in enumerate(self.policy_head.share):
                if isinstance(orig_linear, nn.Linear):
                    iscil_linear = L2MLinear(orig_linear, self.pool_size)
                    self.policy_head.share[i] = iscil_linear
            iscil_linear = L2MLinear(self.policy_head.mean_layer, self.pool_size)
            self.policy_head.mean_layer = iscil_linear 
            iscil_linear = L2MLinear(self.policy_head.logstd_layer, self.pool_size)
            self.policy_head.logstd_layer = iscil_linear
            iscil_linear = L2MLinear(self.policy_head.logits_layer, self.pool_size)
            self.policy_head.logits_layer = iscil_linear


    def spatial_encode_first(self, data):
        # 1. encode image
        with torch.no_grad():
            img_encoded_list = {}
            layer_feature_list = {}
            for img_name in self.image_encoders.keys():
                img = data["obs"][img_name] # (B, T, C, H, W)
                B, T = img.shape[:2]
                img_encoded, layer_feature = TensorUtils.time_distributed(
                    img, self.image_encoder_spatial.first_forward_frozen, 
                    layer_index=self.image_encoder_spatial.lora_layers_list[0]-1)
                layer_feature_list[img_name] = layer_feature
                if self.embed_size == self.image_embed_dim:
                    pass
                else:
                    img_encoded = self.img_down_mlp(img_encoded)
                img_encoded_list[img_name] = img_encoded

            # 2. encode task_emb
            text_tokenzied = data["task_emb"]
            text_encoded = self.language_encoder_spatial(text_tokenzied)  # (B, E_clip)
        
        query = text_encoded
        for img_name in self.image_encoders.keys():
            query = torch.cat([query, img_encoded_list[img_name].mean(dim=-2)], dim=-1)

        return query, layer_feature_list # (B, T, num_modality, E)
    
    def spatial_encode_second(self, data, layer_feature_list):
        # 1. encode image
        img_encoded_list = {}
        for img_name in self.image_encoders.keys():
            img = data["obs"][img_name] # (B, T, C, H, W)
            B, T = img.shape[:2]
            img_encoded = TensorUtils.time_distributed(
                layer_feature_list[img_name], 
                self.image_encoder_spatial.second_forward_lora, 
                layer_index=self.image_encoder_spatial.lora_layers_list[0])
            if self.embed_size == self.image_embed_dim:
                pass
            else:
                img_encoded = self.img_down_mlp(img_encoded)
            img_encoded_list[img_name] = img_encoded

        # 2. encode task_emb
        text_tokenzied = data["task_emb"]
        text_encoded = self.language_encoder_spatial(text_tokenzied)  # (B, E_clip)
        
        # 3. encode extra
        extra = self.extra_encoder(data["obs"])  # (B, T, num_extra, E)
        output = extra

        for img_name in self.image_encoders.keys():
            output = torch.cat([output, img_encoded_list[img_name].unsqueeze(dim=-2)], dim=-2)  # (B, T, num_modality, E)

        # 4. film
        beta, gamma = torch.split(self.fusion_module(text_encoded).reshape(B, self.embed_size * 2), [self.embed_size, self.embed_size], -1)
        beta_all = beta.view(B, 1, 1, self.embed_size).expand(-1, T, self.num_of_modality, -1)
        gamma_all = gamma.view(B, 1, 1, self.embed_size).expand(-1, T, self.num_of_modality, -1)

        output = (1 + gamma_all) * output + beta_all

        return output

    def forward(self, data, frozen_only=False):
        for i, layer in enumerate(self.image_encoder_spatial.model.blocks):
            if isinstance(layer.attn.qkv, L2MLoRAqkv):
                self.image_encoder_spatial.model.blocks[i].attn.qkv.set_idx(None)
        for i, layer in enumerate(self.language_encoder_spatial.model.encoder.layers):
            if isinstance(layer.self_attn.q_proj, L2MLoRA):
                self.language_encoder_spatial.model.encoder.layers[i].self_attn.q_proj.set_idx(None)
            if isinstance(layer.self_attn.q_proj, L2MLoRA):
                self.language_encoder_spatial.model.encoder.layers[i].self_attn.v_proj.set_idx(None)
        for i, layer in enumerate(self.temporal_transformer.layers):
            if isinstance(layer[1].qkv, L2MLoRAqkv):
                self.temporal_transformer.layers[i][1].qkv.set_idx(None)
        
        x_embed, layer_feature_list = self.spatial_encode_first(data)  
        
        prompt_mask = None
        if self.task_id is not None and self.n_tasks is not None:
            if self.task_to_mask is None:
                self.setup_task_mask(x_embed.device)
            prompt_mask = self.task_to_mask[self.task_id]
        
        similarity = self.compute_similarity(x_embed)
        if prompt_mask is not None: 
            similarity[:, ~prompt_mask] = float('-inf')
        
        # take indices of most matched prompt keys --> extract prompt values for respective indices
        _, idx = torch.topk(similarity, k=1, dim=1)  # B, top_k

        if frozen_only:
            return x_embed, idx
        
        for i, layer in enumerate(self.image_encoder_spatial.model.blocks):
            if isinstance(layer.attn.qkv, L2MLoRAqkv):
                self.image_encoder_spatial.model.blocks[i].attn.qkv.set_idx(idx)
        for i, layer in enumerate(self.language_encoder_spatial.model.encoder.layers):
            if isinstance(layer.self_attn.q_proj, L2MLoRA):
                self.language_encoder_spatial.model.encoder.layers[i].self_attn.q_proj.set_idx(idx)
            if isinstance(layer.self_attn.q_proj, L2MLoRA):
                self.language_encoder_spatial.model.encoder.layers[i].self_attn.v_proj.set_idx(idx)
        for i, layer in enumerate(self.temporal_transformer.layers):
            if isinstance(layer[1].qkv, L2MLoRAqkv):
                self.temporal_transformer.layers[i][1].qkv.set_idx(idx)
        
        if not self.lora_only:
            for i, encoder in enumerate(self.extra_encoder.encoders):
                for j, linear in enumerate(encoder):
                    if isinstance(linear, L2MLinear):
                        self.extra_encoder.encoders[i][j].set_idx(idx)
            for i, linear in enumerate(self.fusion_module):
                if isinstance(linear, L2MLinear):
                    self.fusion_module[i].set_idx(idx)
            for i, linear in enumerate(self.policy_head.share):
                if isinstance(linear, L2MLinear):
                    self.policy_head.share[i].set_idx(idx)
            if isinstance(self.policy_head.mean_layer, L2MLinear):
                self.policy_head.mean_layer.set_idx(idx)
            if isinstance(self.policy_head.logstd_layer, L2MLinear):
                self.policy_head.logstd_layer.set_idx(idx)
            if isinstance(self.policy_head.logits_layer, L2MLinear):
                self.policy_head.logits_layer.set_idx(idx)
        
        x = self.spatial_encode_second(data, layer_feature_list)
        x = self.temporal_encode(x)  # second forward pass with lora
        dist = self.policy_head(x)
        
        return dist

    def get_action(self, data):
        self.eval()
        with torch.no_grad():
            with amp.autocast('cuda', dtype=torch.float16):
                data = self.preprocess_input(data, train_mode=False)
                for i, layer in enumerate(self.image_encoder_spatial.model.blocks):
                    if isinstance(layer.attn.qkv, L2MLoRAqkv):
                        self.image_encoder_spatial.model.blocks[i].attn.qkv.set_idx(None)
                for i, layer in enumerate(self.language_encoder_spatial.model.encoder.layers):
                    if isinstance(layer.self_attn.q_proj, L2MLoRA):
                        self.language_encoder_spatial.model.encoder.layers[i].self_attn.q_proj.set_idx(None)
                    if isinstance(layer.self_attn.q_proj, L2MLoRA):
                        self.language_encoder_spatial.model.encoder.layers[i].self_attn.v_proj.set_idx(None)
                for i, layer in enumerate(self.temporal_transformer.layers):
                    if isinstance(layer[1].qkv, L2MLoRAqkv):
                        self.temporal_transformer.layers[i][1].qkv.set_idx(None)
                
                x_embed, layer_feature_list = self.spatial_encode_first(data)
                prompt_mask = None
                if self.task_id is not None and self.n_tasks is not None:
                    if self.task_to_mask is None:
                        self.setup_task_mask(x_embed.device)
                    prompt_mask = self.task_to_mask[self.task_id]
                
                similarity = self.compute_similarity(x_embed)
                if prompt_mask is not None: 
                    similarity[:, ~prompt_mask] = float('-inf')
                
                # take indices of most matched prompt keys --> extract prompt values for respective indices
                _, idx = torch.topk(similarity, k=1, dim=1)  # B, top_k

                for i, layer in enumerate(self.image_encoder_spatial.model.blocks):
                    if isinstance(layer.attn.qkv, L2MLoRAqkv):
                        self.image_encoder_spatial.model.blocks[i].attn.qkv.set_idx(idx)
                for i, layer in enumerate(self.language_encoder_spatial.model.encoder.layers):
                    if isinstance(layer.self_attn.q_proj, L2MLoRA):
                        self.language_encoder_spatial.model.encoder.layers[i].self_attn.q_proj.set_idx(idx)
                    if isinstance(layer.self_attn.q_proj, L2MLoRA):
                        self.language_encoder_spatial.model.encoder.layers[i].self_attn.v_proj.set_idx(idx)
                for i, layer in enumerate(self.temporal_transformer.layers):
                    if isinstance(layer[1].qkv, L2MLoRAqkv):
                        self.temporal_transformer.layers[i][1].qkv.set_idx(idx)

                if not self.lora_only:
                    for i, encoder in enumerate(self.extra_encoder.encoders):
                        for j, linear in enumerate(encoder):
                            if isinstance(linear, L2MLinear):
                                self.extra_encoder.encoders[i][j].set_idx(idx)
                    for i, linear in enumerate(self.fusion_module):
                        if isinstance(linear, L2MLinear):
                            self.fusion_module[i].set_idx(idx)
                    for i, linear in enumerate(self.policy_head.share):
                        if isinstance(linear, L2MLinear):
                            self.policy_head.share[i].set_idx(idx)
                    if isinstance(self.policy_head.mean_layer, L2MLinear):
                        self.policy_head.mean_layer.set_idx(idx)
                    if isinstance(self.policy_head.logstd_layer, L2MLinear):
                        self.policy_head.logstd_layer.set_idx(idx)
                    if isinstance(self.policy_head.logits_layer, L2MLinear):
                        self.policy_head.logits_layer.set_idx(idx)

                x = self.spatial_encode_second(data, layer_feature_list)
                self.latent_queue.append(x)
                if len(self.latent_queue) > self.max_seq_len:
                    self.latent_queue.pop(0)
                x = torch.cat(self.latent_queue, dim=1)  # (B, T, H_all)
                
                x = self.temporal_encode(x)
                dist = self.policy_head(x[:, -1])
        action = dist.sample().detach().cpu()
        
        return action.view(action.shape[0], -1).numpy()
    
    def compute_loss(self, data, reduction="mean"):
        data = self.preprocess_input(data, train_mode=True)
        dist = self.forward(data)
        loss = self.policy_head.loss_fn(dist, data["actions"], reduction)
        return loss
    
    def compute_similarity(self, x_embed):
        # ensure similarity computation does not happen in fp16
        with amp.autocast('cuda', dtype=torch.float32):
            prompt_norm = self.l2_normalize(self.multifacet_prototype, dim=-1)
            x_embed_norm = self.l2_normalize(x_embed, dim=-1)
            similarity_bases = torch.einsum('ije,be->bij', prompt_norm, x_embed_norm)  # B, Pool_size, base_per_proto
            similarity_protos, _ = torch.max(similarity_bases, dim=2)  # B, Pool_size
            if self.pos_sim:
                similarity_protos = (similarity_protos + 1) / 2
        return similarity_protos
    
    @staticmethod
    def l2_normalize(x, dim=None, epsilon=1e-12):
        return torch.nn.functional.normalize(x, p=2.0, dim=dim, eps=epsilon)

    def set_task_id(self, task_id):
        self.task_id = task_id

    def aggregate_embeds(self, x_embed, mask=None):
        if mask is not None:
            # masked mean
            x_embed_mean = torch.sum(x_embed * mask.float().unsqueeze(-1), dim=1) \
                            / torch.sum(mask.float(), -1, keepdim=True)
        else:
            x_embed_mean = torch.mean(x_embed, dim=1)
        return x_embed_mean
    
    def setup_task_mask(self, device):
        self.task_to_mask = {}
        prompt_idx = torch.arange(self.pool_size, device=device)
        previous_ids = []
        for i, ids, in enumerate(prompt_idx.split(self.pool_size // self.n_tasks)):
            previous_ids.append(ids)
            not_mask_ids = torch.cat(previous_ids)
            self.task_to_mask[i] = torch.isin(torch.arange(self.pool_size, device=device), not_mask_ids)
