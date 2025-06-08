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
from libero.lifelong.models.modules.adapter import *
from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.bc_foundation_tail_policy import BCFoundationTailPolicy
from libero.lifelong.models.policy_head import *
from libero.lifelong.models.bc_transformer_policy import ExtraModalityTokens

import time


def reshape_transform(tensor, h, w):
    B, _, E = tensor.shape
    result = tensor[:, 1 : 1 + h * w, :].reshape(B, h, w, E)
    return result.permute(0, 3, 1, 2)

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class BCFoundationDmpelPolicy(BCFoundationTailPolicy):
    """
    Input: (o_{t-H}, ... , o_t)
    Output: a_t or distribution of a_t
    """

    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy
        self.policy_cfg = policy_cfg
        img_encoder_kwargs = policy_cfg.image_encoder.network_kwargs
        img_encoder_kwargs.lora_cfg = policy_cfg.image_encoder.adapter
        text_encoder_kwargs = policy_cfg.language_encoder.network_kwargs
        text_encoder_kwargs.lora_cfg = policy_cfg.language_encoder.adapter
        self.embed_size = policy_cfg.embed_size
        self.has_pretrained = policy_cfg.has_pretrained

        ### 1. encode image
        self.image_encoders = {}
        
        for name in shape_meta["all_shapes"].keys():
            if "rgb" in name or "depth" in name:
                self.image_encoders[name] = {
                    "input_shape": shape_meta["all_shapes"][name],
                }
        self.num_of_image = len(self.image_encoders.keys())

        self.image_encoder_spatial = eval(policy_cfg.image_encoder.network)(**img_encoder_kwargs)
        self.image_embed_dim = self.image_encoder_spatial.model.embed_dim
        
        ### 2. encode language
        self.language_encoder_spatial = eval(policy_cfg.language_encoder.network)(**text_encoder_kwargs)
        self.language_embed_dim = self.language_encoder_spatial.model.config.hidden_size
        self.frozen_language_emb = None

        ### 5. encode extra information (e.g. gripper, joint_state)
        self.extra_encoder = ExtraModalityTokens(
            use_joint=cfg.data.use_joint,
            use_gripper=cfg.data.use_gripper,
            use_ee=cfg.data.use_ee,
            extra_num_layers=policy_cfg.extra_num_layers,
            extra_hidden_size=policy_cfg.extra_hidden_size,
            extra_embedding_size=self.embed_size,
        )
        self.joint_states_dim = 7
        self.gripper_states_dim = 2
        self.ee_dim = 3
        
        ### 6. FiLM
        if self.embed_size == self.image_embed_dim:
            pass
        else:
            self.img_down_mlp = nn.Linear(self.image_embed_dim, self.embed_size)
        
        self.num_of_extra = self.extra_encoder.num_extra
        
        self.num_of_modality = self.num_of_image + self.num_of_extra
        self.fusion_module = nn.Sequential(nn.Linear(self.language_embed_dim, policy_cfg.film_hidden_size),
                                            nn.GELU('tanh'),
                                            nn.Linear(policy_cfg.film_hidden_size, self.embed_size * 2),
                                            )
        
        ### 7. define temporal transformer
        policy_cfg.temporal_position_encoding.network_kwargs.input_size = self.embed_size
        self.temporal_transformer_position_encoding_fn = eval(
            policy_cfg.temporal_position_encoding.network
        )(**policy_cfg.temporal_position_encoding.network_kwargs)

        self.temporal_transformer = TransformerDecoder(
            input_size=self.embed_size,
            num_layers=policy_cfg.transformer_num_layers,
            num_heads=policy_cfg.transformer_num_heads,
            head_output_size=policy_cfg.transformer_head_output_size,
            mlp_hidden_size=policy_cfg.transformer_mlp_hidden_size,
            dropout=policy_cfg.transformer_dropout,
            use_lora=policy_cfg.use_lora,
            fullft=policy_cfg.fullft,
            lora_cfg=policy_cfg.adapter,
        )

        policy_head_kwargs = policy_cfg.policy_head.network_kwargs
        policy_head_kwargs.input_size = self.embed_size
        policy_head_kwargs.output_size = shape_meta["ac_dim"]

        self.policy_head = eval(policy_cfg.policy_head.network)(
            **policy_cfg.policy_head.loss_kwargs,
            **policy_cfg.policy_head.network_kwargs
        )
        self.latent_queue = []
        self.max_seq_len = policy_cfg.transformer_max_seq_len

        ### 8. reshape transform for attention visualization
        self.reshape_transform = lambda x: reshape_transform(
            x, self.image_encoder_spatial[0].h, self.image_encoder_spatial[1].w
        )
        
    def init_moe_policy(self):
        for k, v in self.named_parameters():
            v.requires_grad = False
        self.task_emb_size = self.policy_cfg.task_emb_size
        self.pool_size = self.policy_cfg.init_pool_size
        self.ll_expert_per_task = self.policy_cfg.ll_expert_per_task
        self.query_use_proprio = self.policy_cfg.query_use_proprio
        self.router_coeff_seperate = self.policy_cfg.router_coeff_seperate
        self.infer_interval = self.policy_cfg.infer_interval
        self.query_use_mean_img = self.policy_cfg.query_use_mean_img
        self.query_use_diff_img = self.policy_cfg.query_use_diff_img
        router_in_dim = self.language_embed_dim
        if self.query_use_mean_img:
            router_in_dim += self.image_embed_dim * 2
        if self.query_use_diff_img:
            router_in_dim += self.image_embed_dim * 2
        if self.query_use_proprio:
            router_in_dim += self.joint_states_dim + self.gripper_states_dim
        self.moe_router = MoERouterCoeff(in_dim=router_in_dim,
                                        hidden_dim=self.policy_cfg.task_emb_net_hidden_size,
                                        cfg=self.policy_cfg)
        self.moe_topk = self.policy_cfg.moe_topk
    
        lora_kwargs = {'rank': self.policy_cfg.adapter.rank,
                        'init_pool_size': self.pool_size,
                        'merge_AB': self.policy_cfg.merge_AB,
                        'tune_bias': self.policy_cfg.tune_bias,
                        'lora_dropout': self.policy_cfg.lora_dropout,
                        }
        for i, encoder in enumerate(self.extra_encoder.encoders):
            for j, orig_linear in enumerate(encoder):
                if isinstance(orig_linear, nn.Linear):
                    moe_linear = MoELoRA(orig_linear, **lora_kwargs)
                    self.extra_encoder.encoders[i][j] = moe_linear
        for i, orig_linear in enumerate(self.fusion_module):
            if isinstance(orig_linear, nn.Linear):
                moe_linear = MoELoRA(orig_linear, **lora_kwargs)
                self.fusion_module[i] = moe_linear
        for i, orig_linear in enumerate(self.policy_head.share):
            if isinstance(orig_linear, nn.Linear):
                moe_linear = MoELoRA(orig_linear, **lora_kwargs)
                self.policy_head.share[i] = moe_linear
        moe_linear = MoELoRA(self.policy_head.mean_layer, **lora_kwargs)
        self.policy_head.mean_layer = moe_linear 
        moe_linear = MoELoRA(self.policy_head.logstd_layer, **lora_kwargs)
        self.policy_head.logstd_layer = moe_linear 
        moe_linear = MoELoRA(self.policy_head.logits_layer, **lora_kwargs)
        self.policy_head.logits_layer = moe_linear
        self.use_t_lora = self.policy_cfg.t_lora
        self.use_s_lora_image = self.policy_cfg.s_lora_image
        self.use_s_lora_text = self.policy_cfg.s_lora_text
        if self.policy_cfg.t_lora:
            if self.policy_cfg.t_lora_layer_list == "all":
                self.t_lora_layer_list = list(range(len(self.temporal_transformer.layers)))
            else:
                self.t_lora_layer_list = self.policy_cfg.t_lora_layer_list
            assert isinstance(self.t_lora_layer_list, list)
            for i, layer in enumerate(self.temporal_transformer.layers):
                if i in self.t_lora_layer_list:
                    orig_qkv = layer[1].qkv
                    qkv_lora = MoELoRAqkv(orig_qkv, **lora_kwargs)
                    layer[1].qkv = qkv_lora
        lora_kwargs['rank'] = int(self.policy_cfg.adapter.rank/2)
        lora_kwargs['tune_bias'] = False
        if self.policy_cfg.s_lora_image:
            if self.policy_cfg.s_lora_image_layer_list == "all":
                self.s_lora_image_layer_list = list(range(len(self.image_encoder_spatial.model.blocks)))
            else:
                self.s_lora_image_layer_list = self.policy_cfg.s_lora_image_layer_list
            assert isinstance(self.s_lora_image_layer_list, list)
            for i, layer in enumerate(self.image_encoder_spatial.model.blocks):
                if i in self.s_lora_image_layer_list:
                    orig_qkv = layer.attn.qkv
                    qkv_lora = MoELoRAqkv(orig_qkv, **lora_kwargs)
                    self.image_encoder_spatial.model.blocks[i].attn.qkv = qkv_lora
        if self.policy_cfg.s_lora_text:
            if self.policy_cfg.s_lora_text_layer_list == "all":
                self.s_lora_text_layer_list = list(range(len(self.language_encoder_spatial.model.encoder.layers)))
            else:
                self.s_lora_text_layer_list = self.policy_cfg.s_lora_text_layer_list
            assert isinstance(self.s_lora_text_layer_list, list)
            for i, layer in enumerate(self.language_encoder_spatial.model.encoder.layers):
                if i in self.s_lora_text_layer_list:
                    orig_q = layer.self_attn.q_proj
                    orig_v = layer.self_attn.v_proj
                    q_lora = MoELoRA(orig_q, **lora_kwargs)
                    v_lora = MoELoRA(orig_v, **lora_kwargs)
                    self.language_encoder_spatial.model.encoder.layers[i].self_attn.q_proj = q_lora
                    self.language_encoder_spatial.model.encoder.layers[i].self_attn.v_proj = v_lora
    
    def spatial_encode(self, data, layer_feature_list=None):
        if self.use_s_lora_image:
            for i, layer in enumerate(self.image_encoder_spatial.model.blocks):
                if isinstance(layer.attn.qkv, MoELoRAqkv):
                    self.image_encoder_spatial.model.blocks[i].attn.qkv.set_use_lora(True)
        if self.use_s_lora_text:
            for i, layer in enumerate(self.language_encoder_spatial.model.encoder.layers):
                if isinstance(layer.self_attn.q_proj, MoELoRA):
                    self.language_encoder_spatial.model.encoder.layers[i].self_attn.q_proj.set_use_lora(True)
                if isinstance(layer.self_attn.q_proj, MoELoRA):
                    self.language_encoder_spatial.model.encoder.layers[i].self_attn.v_proj.set_use_lora(True)
        # 1. encode image
        img_encoded_list = {}
        for img_name in self.image_encoders.keys():
            img = data["obs"][img_name] # (B, T, C, H, W)
            B, T = img.shape[:2]
            if layer_feature_list is not None:
                img_encoded = TensorUtils.time_distributed(
                            layer_feature_list[img_name], 
                            self.image_encoder_spatial.second_forward_lora, 
                            layer_index=self.s_lora_image_layer_list[0])
            else:
                img_encoded = TensorUtils.time_distributed(
                            img, self.image_encoder_spatial.forward)
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

        return output # (B, T, num_modality, E)
    
    def context_encode(self, data):
        if self.use_s_lora_image:
            for i, layer in enumerate(self.image_encoder_spatial.model.blocks):
                if isinstance(layer.attn.qkv, MoELoRAqkv):
                    self.image_encoder_spatial.model.blocks[i].attn.qkv.set_use_lora(False)
        if self.use_s_lora_text:
            for i, layer in enumerate(self.language_encoder_spatial.model.encoder.layers):
                if isinstance(layer.self_attn.q_proj, MoELoRA):
                    layer.self_attn.q_proj.set_use_lora(False)
                if isinstance(layer.self_attn.v_proj, MoELoRA):
                    layer.self_attn.v_proj.set_use_lora(False)
        # 1. encode image
        with torch.no_grad():
            img_encoded_list = {}
            layer_feature_list = {}
            for img_name in self.image_encoders.keys():
                img = data["obs"][img_name] # (B, T, C, H, W)
                B, T = img.shape[:2]
                if self.policy_cfg.s_lora_image:
                    img_encoded, layer_feature = TensorUtils.time_distributed(
                        img, self.image_encoder_spatial.first_forward_frozen, 
                        layer_index=self.s_lora_image_layer_list[0]-1)
                    layer_feature_list[img_name] = layer_feature
                if self.embed_size == self.image_embed_dim:
                    pass
                else:
                    img_encoded = self.img_down_mlp(img_encoded)
                img_encoded_list[img_name] = img_encoded
            
            # 2. encode task_emb
            text_tokenzied = data["task_emb"]
            text_encoded = self.language_encoder_spatial(text_tokenzied)

        query_in = text_encoded
        for k, v in img_encoded_list.items():
            if self.query_use_mean_img:
                img_emb_mean = v.mean(dim=-2)
                query_in = torch.cat([query_in, img_emb_mean], dim=-1)
            if self.query_use_diff_img:
                img_emb_diff = v[:,-1,:] - v[:,0,:]
                query_in = torch.cat([query_in, img_emb_diff], dim=-1)
        if self.query_use_proprio:
            for modality_name in ["joint_states", "gripper_states"]:
                query_proprio = data["obs"][modality_name].mean(dim=-2)
                query_in= torch.cat([query_in, query_proprio], dim=-1)
        
        return layer_feature_list, query_in
    
    
    def infer_lora(self, query_in, mode='train'):
        with amp.autocast('cuda', dtype=torch.float32):
            topk_attn_norm, topk_idx = self.moe_router.forward(query_in)
            if mode == 'save_attn':
                return topk_idx, topk_attn_norm

        if self.router_coeff_seperate:
            bsz = topk_attn_norm.shape[0]
            topk_attn_norm_multi = topk_attn_norm.view(bsz, 6, -1)
            topk_idx_multi = topk_idx.view(bsz, 6, -1)
            topk_attn_norm_sep_list = {}
            topk_idx_sep_list = {}
            for i, key in zip(range(6), ['img', 'txt', 'extra', 'fusion', 'tem', 'head']):
                topk_attn_norm_sep_list[key] = topk_attn_norm_multi[:, i, :]
                topk_idx_sep_list[key] = topk_idx_multi[:, i, :]

        for i, encoder in enumerate(self.extra_encoder.encoders):
            for j, linear in enumerate(encoder):
                if isinstance(linear, MoELoRA):
                    if self.router_coeff_seperate:
                        self.extra_encoder.encoders[i][j].set_attentions(topk_attn_norm_sep_list['extra'], topk_idx_sep_list['extra'])
                    else:
                        self.extra_encoder.encoders[i][j].set_attentions(topk_attn_norm, topk_idx)
        for i, linear in enumerate(self.fusion_module):
            if isinstance(linear, MoELoRA):
                if self.router_coeff_seperate:
                    self.fusion_module[i].set_attentions(topk_attn_norm_sep_list['fusion'], topk_idx_sep_list['fusion'])
                else:
                    self.fusion_module[i].set_attentions(topk_attn_norm, topk_idx)
        for i, linear in enumerate(self.policy_head.share):
            if isinstance(linear, MoELoRA):
                if self.router_coeff_seperate:
                    self.policy_head.share[i].set_attentions(topk_attn_norm_sep_list['head'], topk_idx_sep_list['head'])
                else:
                    self.policy_head.share[i].set_attentions(topk_attn_norm, topk_idx)
        if isinstance(linear, MoELoRA):
            if self.router_coeff_seperate:
                self.policy_head.mean_layer.set_attentions(topk_attn_norm_sep_list['head'], topk_idx_sep_list['head'])
            else:
                self.policy_head.mean_layer.set_attentions(topk_attn_norm, topk_idx)
        if isinstance(linear, MoELoRA):
            if self.router_coeff_seperate:
                self.policy_head.logstd_layer.set_attentions(topk_attn_norm_sep_list['head'], topk_idx_sep_list['head'])
            else:
                self.policy_head.logstd_layer.set_attentions(topk_attn_norm, topk_idx)
        if isinstance(linear, MoELoRA):
            if self.router_coeff_seperate:
                self.policy_head.logits_layer.set_attentions(topk_attn_norm_sep_list['head'], topk_idx_sep_list['head'])
            else:
                self.policy_head.logits_layer.set_attentions(topk_attn_norm, topk_idx)
        if self.use_t_lora:
            for i, layer in enumerate(self.temporal_transformer.layers):
                if isinstance(self.temporal_transformer.layers[i][1].qkv, MoELoRAqkv):
                    if self.router_coeff_seperate:
                        self.temporal_transformer.layers[i][1].qkv.set_attentions(topk_attn_norm_sep_list['tem'], topk_idx_sep_list['tem'])
                    else:
                        self.temporal_transformer.layers[i][1].qkv.set_attentions(topk_attn_norm, topk_idx)
                    layer[1].qkv.set_use_lora(True)
                        
        if self.use_s_lora_image:
            for i, layer in enumerate(self.image_encoder_spatial.model.blocks):
                if isinstance(layer.attn.qkv, MoELoRAqkv):
                    if self.router_coeff_seperate:
                        layer.attn.qkv.set_attentions(topk_attn_norm_sep_list['img'], topk_idx_sep_list['img'])
                    else:
                        layer.attn.qkv.set_attentions(topk_attn_norm, topk_idx)
                    layer.attn.qkv.set_use_lora(True)
        
        if self.use_s_lora_text:
            for i, layer in enumerate(self.language_encoder_spatial.model.encoder.layers):
                if isinstance(layer.self_attn.q_proj, MoELoRA):
                    if self.router_coeff_seperate:
                        layer.self_attn.q_proj.set_attentions(topk_attn_norm_sep_list['txt'], topk_idx_sep_list['txt'])
                    else:
                        layer.self_attn.q_proj.set_attentions(topk_attn_norm, topk_idx)
                    layer.self_attn.q_proj.set_use_lora(True)
                if isinstance(layer.self_attn.v_proj, MoELoRA):
                    if self.router_coeff_seperate:
                        layer.self_attn.v_proj.set_attentions(topk_attn_norm_sep_list['txt'], topk_idx_sep_list['txt'])
                    else:
                        layer.self_attn.v_proj.set_attentions(topk_attn_norm, topk_idx)
                    layer.self_attn.v_proj.set_use_lora(True)
        
        return topk_idx, topk_attn_norm
        
    def temporal_encode(self, x):
        pos_emb = self.temporal_transformer_position_encoding_fn(x)
        x = x + pos_emb.unsqueeze(1)  # (B, T, num_modality, E)
        sh = x.shape
        self.temporal_transformer.compute_mask(x.shape)

        x = TensorUtils.join_dimensions(x, 1, 2)  # (B, T*num_modality, E)
        x = self.temporal_transformer(x)
        x = x.reshape(*sh)
        return x[:, :, 0]  # (B, T, E)

    def forward(self, data):
        layer_feature_list, query_in = self.context_encode(data)
        topk_idx, topk_attn_norm = self.infer_lora(query_in)
        x = self.spatial_encode(data, layer_feature_list)  # (B, T, num_modality, E)
        x = self.temporal_encode(x)  # (B, T, E)
        dist = self.policy_head(x)
        return dist, query_in, topk_idx, topk_attn_norm

    def compute_loss(self, data, reduction="mean"):
        data = self.preprocess_input(data, train_mode=True)
        dist, query_in, topk_idx, topk_attn_norm = self.forward(data)
        bc_loss = self.policy_head.loss_fn(dist, data["actions"], reduction)
        # loss = bc_loss
        return bc_loss
    
    def get_action(self, data):
        self.eval()
        with torch.no_grad():
            with amp.autocast('cuda', dtype=torch.float16):
                data = self.preprocess_input(data, train_mode=False)
                if self.infer_flag % self.infer_interval == 0:
                    layer_feature_list, query_in = self.context_encode(data)
                    self.context_queue.append(query_in)
                    if len(self.context_queue) > self.max_seq_len:
                        self.context_queue.pop(0)
                    query_in = torch.stack(self.context_queue, dim=0).mean(dim=0)  # (B, T, H_all)
                    topk_idx, topk_attn_norm = self.infer_lora(query_in)
                    x = self.spatial_encode(data, layer_feature_list)
                else:
                    if self.query_use_proprio:
                        query_in = self.context_queue[-1][...,:-(self.joint_states_dim+self.gripper_states_dim)]
                        for modality_name in ["joint_states", "gripper_states"]:
                            query_proprio = data["obs"][modality_name].mean(dim=-2)
                            query_in= torch.cat([query_in, query_proprio], dim=-1)
                    else:
                        query_in = self.context_queue[-1]
                    self.context_queue.append(query_in)
                    if len(self.context_queue) > self.max_seq_len:
                        self.context_queue.pop(0)
                    x = self.spatial_encode(data)
                self.latent_queue.append(x)
                if len(self.latent_queue) > self.max_seq_len:
                    self.latent_queue.pop(0)
                x = torch.cat(self.latent_queue, dim=1)  # (B, T, H_all)
                x = self.temporal_encode(x)
                dist = self.policy_head(x[:, -1])
        action = dist.sample().detach().cpu()
        self.infer_flag += 1
        return action.view(action.shape[0], -1).numpy()

    def recall_moe_attention(self, replayed_query_in, replayed_attn_vector):
        bsz = replayed_query_in.shape[0]
        now_topk_attn_norm, now_topk_idx = self.moe_router.forward(replayed_query_in)
        if self.router_coeff_seperate:
            topk_attn_norm_multi = now_topk_attn_norm.view(bsz, 6, -1)
            topk_idx_multi = now_topk_idx.view(bsz, 6, -1)
            now_attn_vector = torch.zeros((bsz, 6, self.pool_size), device=now_topk_attn_norm.device)
            for i in range(6):
                now_attn_vector[:, i, :].scatter_(1, topk_idx_multi[:, i, :], topk_attn_norm_multi[:, i, :])
            now_attn_vector.view(bsz, -1)
        else:
            now_attn_vector = torch.zeros((bsz, self.pool_size), device=now_topk_attn_norm.device)
            now_attn_vector.scatter_(1, now_topk_idx, now_topk_attn_norm)
        replayed_attn_loss = torch.nn.functional.mse_loss(now_attn_vector, replayed_attn_vector)
        return replayed_attn_loss, now_attn_vector

    def reset(self):
        self.context_queue = []
        self.latent_queue = []
        self.infer_flag = 0

    def add_new_and_freeze_previous(self, add_expert_num):
        self.pool_size += add_expert_num
        self.moe_router.add_expert(add_expert_num) # if new_task_mean, reinit in the algo 
        for i, encoder in enumerate(self.extra_encoder.encoders):
            for j, linear in enumerate(encoder):
                if isinstance(linear, MoELoRA):
                    self.extra_encoder.encoders[i][j].add_expert(add_expert_num, self.cfg.policy.ll_expert_init)
        for i, linear in enumerate(self.fusion_module):
            if isinstance(linear, MoELoRA):
                self.fusion_module[i].add_expert(add_expert_num, self.cfg.policy.ll_expert_init)
        for i, linear in enumerate(self.policy_head.share):
            if isinstance(linear, MoELoRA):
                self.policy_head.share[i].add_expert(add_expert_num, self.cfg.policy.ll_expert_init)
        if isinstance(self.policy_head.mean_layer, MoELoRA):
            self.policy_head.mean_layer.add_expert(add_expert_num, self.cfg.policy.ll_expert_init)
        if isinstance(self.policy_head.logstd_layer, MoELoRA):
            self.policy_head.logstd_layer.add_expert(add_expert_num, self.cfg.policy.ll_expert_init)
        if isinstance(self.policy_head.logits_layer, MoELoRA):
            self.policy_head.logits_layer.add_expert(add_expert_num, self.cfg.policy.ll_expert_init)
        if self.use_t_lora:
            for i, layer in enumerate(self.temporal_transformer.layers):
                if isinstance(self.temporal_transformer.layers[i][1].qkv, MoELoRAqkv):
                    self.temporal_transformer.layers[i][1].qkv.add_expert(add_expert_num, self.cfg.policy.ll_expert_init)
        if self.use_s_lora_image:
            for i, layer in enumerate(self.image_encoder_spatial.model.blocks):
                if isinstance(layer.attn.qkv, MoELoRAqkv):
                    self.image_encoder_spatial.model.blocks[i].attn.qkv.add_expert(add_expert_num, self.cfg.policy.ll_expert_init)
        if self.use_s_lora_text:
            for i, layer in enumerate(self.language_encoder_spatial.model.encoder.layers):
                if isinstance(layer.self_attn.q_proj, MoELoRA):
                    self.language_encoder_spatial.model.encoder.layers[i].self_attn.q_proj.add_expert(add_expert_num, self.cfg.policy.ll_expert_init)
                if isinstance(layer.self_attn.v_proj, MoELoRA):
                    self.language_encoder_spatial.model.encoder.layers[i].self_attn.v_proj.add_expert(add_expert_num, self.cfg.policy.ll_expert_init)
        # for name, tensor in self.named_parameters():
        #     if tensor.requires_grad:
        #         print('{}: {}, {}'.format(name, torch.numel(tensor), tensor.requires_grad))
        # total_params = sum(p.numel() for p in self.parameters())
        # print("Total parameters: {}".format(total_params))
        # trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # print("Trainable parameters: {}".format(trainable_params))
