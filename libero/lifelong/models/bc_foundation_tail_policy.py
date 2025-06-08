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
from libero.lifelong.models.policy_head import *
from libero.lifelong.models.bc_transformer_policy import ExtraModalityTokens



def reshape_transform(tensor, h, w):
    B, _, E = tensor.shape
    result = tensor[:, 1 : 1 + h * w, :].reshape(B, h, w, E)
    return result.permute(0, 3, 1, 2)

def _convert_image_to_rgb(image):
    return image.convert("RGB")


class BCFoundationTailPolicy(BasePolicy):
    """
    Input: (o_{t-H}, ... , o_t)
    Output: a_t or distribution of a_t
    """

    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy
        img_encoder_kwargs = policy_cfg.image_encoder.network_kwargs
        img_encoder_kwargs.lora_cfg = policy_cfg.image_encoder.adapter
        text_encoder_kwargs = policy_cfg.language_encoder.network_kwargs
        text_encoder_kwargs.lora_cfg = policy_cfg.language_encoder.adapter
        # self.shape_meta = shape_meta
        self.embed_size = policy_cfg.embed_size

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

        ### 5. encode extra information (e.g. gripper, joint_state)
        self.extra_encoder = ExtraModalityTokens(
            use_joint=cfg.data.use_joint,
            use_gripper=cfg.data.use_gripper,
            use_ee=cfg.data.use_ee,
            extra_num_layers=policy_cfg.extra_num_layers,
            extra_hidden_size=policy_cfg.extra_hidden_size,
            extra_embedding_size=self.embed_size,
        )

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
        

    def spatial_encode(self, data):
        # 1. encode image
        img_encoded_list = {}
        for img_name in self.image_encoders.keys():
            img = data["obs"][img_name] # (B, T, C, H, W)
            B, T = img.shape[:2]
            img_encoded = TensorUtils.time_distributed(img, self.image_encoder_spatial)
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
        x = self.spatial_encode(data)  # (B, T, num_modality, E)
        x = self.temporal_encode(x)  # (B, T, E)
        dist = self.policy_head(x)
        return dist

    def get_action(self, data):
        self.eval()
        with torch.no_grad():
            with amp.autocast('cuda', dtype=torch.float16):
                data = self.preprocess_input(data, train_mode=False)
                x = self.spatial_encode(data)
                self.latent_queue.append(x)
                if len(self.latent_queue) > self.max_seq_len:
                    self.latent_queue.pop(0)
                x = torch.cat(self.latent_queue, dim=1)  # (B, T, H_all)
                x = self.temporal_encode(x)
                dist = self.policy_head(x[:, -1])
        action = dist.sample().detach().cpu()
        return action.view(action.shape[0], -1).numpy()

    def reset(self):
        self.latent_queue = []