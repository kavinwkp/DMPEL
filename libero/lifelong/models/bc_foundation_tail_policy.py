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
from libero.lifelong.models.AnalyticLinear import ACIL, DSAL
from libero.lifelong.models.policy_head import *
from libero.lifelong.models.bc_transformer_policy import ExtraModalityTokens
from libero.lifelong.models.diffusion_head import DiffusionPolicy


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
        self.embed_size = policy_cfg.embed_size     # 768

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
            num_layers=policy_cfg.transformer_num_layers,   # 6
            num_heads=policy_cfg.transformer_num_heads,     # 8
            head_output_size=policy_cfg.transformer_head_output_size,   # 96
            mlp_hidden_size=policy_cfg.transformer_mlp_hidden_size,     # 1024
            dropout=policy_cfg.transformer_dropout, # 0.15
            use_lora=policy_cfg.use_lora,   # LoRAqkv
            fullft=policy_cfg.fullft,       # false
            lora_cfg=policy_cfg.adapter,    # lora16
        )

        # policy_head_kwargs = policy_cfg.policy_head.network_kwargs
        # policy_head_kwargs.input_size = self.embed_size
        # policy_head_kwargs.output_size = shape_meta["ac_dim"]
        #
        # self.policy_head = eval(policy_cfg.policy_head.network)(
        #     **policy_cfg.policy_head.loss_kwargs,
        #     **policy_cfg.policy_head.network_kwargs
        # )

        self.num_queries = cfg.data.seq_len
        self.step = 0

        # self.policy_head = nn.Linear(self.embed_size, shape_meta["ac_dim"] * self.num_queries)

        self.policy_head = DiffusionPolicy(obs_dim=self.embed_size,
                                           act_dim=shape_meta["ac_dim"],
                                           obs_horizon=10,
                                           pred_horizon=cfg.data.seq_len,
                                           hidden_dim=self.embed_size,
                                           num_layers=2,
                                           policy_type="transformer",
                                           device=cfg.device)
        self.all_time_actions = torch.zeros((20, 600, 600 + self.num_queries, shape_meta["ac_dim"])).to(cfg.device)

        self.latent_queue = []
        self.max_seq_len = policy_cfg.transformer_max_seq_len

        ### 8. reshape transform for attention visualization
        self.reshape_transform = lambda x: reshape_transform(
            x, self.image_encoder_spatial[0].h, self.image_encoder_spatial[1].w
        )

        # for param in self.extra_encoder.parameters():
        #     param.requires_grad = False
        #
        # for param in self.fusion_module.parameters():
        #     param.requires_grad = False
        #
        # for param in self.policy_head.parameters():
        #     param.requires_grad = False

    # def init_policy_head(self):
    #     self.policy_head = ACIL(
    #         backbone_output_size=768,
    #         buffer_size=8192,
    #         out_features=350,
    #         gamma=0.1,
    #         device=self.cfg.device,
    #         dtype=torch.double)
    #
    #     # self.policy_head = DSAL(
    #     #     backbone_output_size=768,
    #     #     buffer_size=8192,
    #     #     out_features=350,
    #     #     device=self.cfg.device,
    #     #     dtype=torch.double)
    #
    #     print(self.policy_head)

    def init_lora(self):
        self.image_encoder_spatial.init_lora()
        self.language_encoder_spatial.init_lora()
        self.temporal_transformer.init_lora()

        self.lora_rank = 16

        for i, encoder in enumerate(self.extra_encoder.encoders):
            for j, orig_linear in enumerate(encoder):
                if isinstance(orig_linear, nn.Linear):
                    moe_linear = LoRA(orig_linear, dim=orig_linear.in_features, rank=self.lora_rank, dim_out=orig_linear.out_features)
                    self.extra_encoder.encoders[i][j] = moe_linear

        for i, orig_linear in enumerate(self.fusion_module):
            if isinstance(orig_linear, nn.Linear):
                moe_linear = LoRA(orig_linear, dim=orig_linear.in_features, rank=self.lora_rank, dim_out=orig_linear.out_features)
                self.fusion_module[i] = moe_linear

        # moe_linear = LoRA(self.policy_head.noise_pred_net.encoder[0], dim=self.policy_head.noise_pred_net.encoder[0].in_features, rank=self.lora_rank, dim_out=self.policy_head.noise_pred_net.encoder[0].out_features)
        # self.policy_head.noise_pred_net.encoder[0] = moe_linear
        # moe_linear = LoRA(self.policy_head.noise_pred_net.encoder[2], dim=self.policy_head.noise_pred_net.encoder[2].in_features, rank=self.lora_rank, dim_out=self.policy_head.noise_pred_net.encoder[2].out_features)
        # self.policy_head.noise_pred_net.encoder[2] = moe_linear
        #
        # moe_linear = LoRA(self.policy_head.noise_pred_net.decoder.layers[0].self_attn.out_proj, dim=self.policy_head.noise_pred_net.decoder.layers[0].self_attn.out_proj.in_features, rank=self.lora_rank)
        # self.policy_head.noise_pred_net.decoder.layers[0].self_attn.out_proj = moe_linear
        # moe_linear = LoRA(self.policy_head.noise_pred_net.decoder.layers[0].multihead_attn.out_proj, dim=self.policy_head.noise_pred_net.decoder.layers[0].multihead_attn.out_proj.in_features, rank=self.lora_rank)
        # self.policy_head.noise_pred_net.decoder.layers[0].multihead_attn.out_proj = moe_linear
        # moe_linear = LoRA(self.policy_head.noise_pred_net.decoder.layers[1].self_attn.out_proj, dim=self.policy_head.noise_pred_net.decoder.layers[1].self_attn.out_proj.in_features, rank=self.lora_rank)
        # self.policy_head.noise_pred_net.decoder.layers[1].self_attn.out_proj = moe_linear
        # moe_linear = LoRA(self.policy_head.noise_pred_net.decoder.layers[1].multihead_attn.out_proj, dim=self.policy_head.noise_pred_net.decoder.layers[1].multihead_attn.out_proj.in_features, rank=self.lora_rank)
        # self.policy_head.noise_pred_net.decoder.layers[1].multihead_attn.out_proj = moe_linear
        #
        # moe_linear = LoRA(self.policy_head.noise_pred_net.decoder.layers[0].linear1, dim=self.policy_head.noise_pred_net.decoder.layers[0].linear1.in_features, rank=self.lora_rank,
        #                   dim_out=self.policy_head.noise_pred_net.decoder.layers[0].linear1.out_features)
        # self.policy_head.noise_pred_net.decoder.layers[0].linear1 = moe_linear
        # moe_linear = LoRA(self.policy_head.noise_pred_net.decoder.layers[0].linear2, dim=self.policy_head.noise_pred_net.decoder.layers[0].linear2.in_features, rank=self.lora_rank,
        #                   dim_out=self.policy_head.noise_pred_net.decoder.layers[0].linear2.out_features)
        # self.policy_head.noise_pred_net.decoder.layers[0].linear2 = moe_linear
        # moe_linear = LoRA(self.policy_head.noise_pred_net.decoder.layers[1].linear1, dim=self.policy_head.noise_pred_net.decoder.layers[1].linear1.in_features, rank=self.lora_rank,
        #                   dim_out=self.policy_head.noise_pred_net.decoder.layers[1].linear1.out_features)
        # self.policy_head.noise_pred_net.decoder.layers[1].linear1 = moe_linear
        # moe_linear = LoRA(self.policy_head.noise_pred_net.decoder.layers[1].linear2, dim=self.policy_head.noise_pred_net.decoder.layers[1].linear2.in_features, rank=self.lora_rank,
        #                   dim_out=self.policy_head.noise_pred_net.decoder.layers[1].linear2.out_features)
        # self.policy_head.noise_pred_net.decoder.layers[1].linear2 = moe_linear
        # moe_linear = LoRA(self.policy_head.noise_pred_net.head, dim=self.policy_head.noise_pred_net.head.in_features, rank=self.lora_rank, dim_out=self.policy_head.noise_pred_net.head.out_features)
        # self.policy_head.noise_pred_net.head = moe_linear
        #
        # # TODO: ema
        # moe_linear = LoRA(self.policy_head.ema_noise_pred_net.encoder[0], dim=self.policy_head.ema_noise_pred_net.encoder[0].in_features, rank=self.lora_rank, dim_out=self.policy_head.ema_noise_pred_net.encoder[0].out_features)
        # self.policy_head.ema_noise_pred_net.encoder[0] = moe_linear
        # moe_linear = LoRA(self.policy_head.ema_noise_pred_net.encoder[2], dim=self.policy_head.ema_noise_pred_net.encoder[2].in_features, rank=self.lora_rank, dim_out=self.policy_head.ema_noise_pred_net.encoder[2].out_features)
        # self.policy_head.ema_noise_pred_net.encoder[2] = moe_linear
        #
        # moe_linear = LoRA(self.policy_head.ema_noise_pred_net.decoder.layers[0].self_attn.out_proj, dim=self.policy_head.ema_noise_pred_net.decoder.layers[0].self_attn.out_proj.in_features, rank=self.lora_rank)
        # self.policy_head.ema_noise_pred_net.decoder.layers[0].self_attn.out_proj = moe_linear
        # moe_linear = LoRA(self.policy_head.ema_noise_pred_net.decoder.layers[0].multihead_attn.out_proj, dim=self.policy_head.ema_noise_pred_net.decoder.layers[0].multihead_attn.out_proj.in_features, rank=self.lora_rank)
        # self.policy_head.ema_noise_pred_net.decoder.layers[0].multihead_attn.out_proj = moe_linear
        # moe_linear = LoRA(self.policy_head.ema_noise_pred_net.decoder.layers[1].self_attn.out_proj, dim=self.policy_head.ema_noise_pred_net.decoder.layers[1].self_attn.out_proj.in_features, rank=self.lora_rank)
        # self.policy_head.ema_noise_pred_net.decoder.layers[1].self_attn.out_proj = moe_linear
        # moe_linear = LoRA(self.policy_head.ema_noise_pred_net.decoder.layers[1].multihead_attn.out_proj, dim=self.policy_head.ema_noise_pred_net.decoder.layers[1].multihead_attn.out_proj.in_features, rank=self.lora_rank)
        # self.policy_head.ema_noise_pred_net.decoder.layers[1].multihead_attn.out_proj = moe_linear
        #
        # moe_linear = LoRA(self.policy_head.ema_noise_pred_net.decoder.layers[0].linear1, dim=self.policy_head.ema_noise_pred_net.decoder.layers[0].linear1.in_features, rank=self.lora_rank,
        #                   dim_out=self.policy_head.ema_noise_pred_net.decoder.layers[0].linear1.out_features)
        # self.policy_head.ema_noise_pred_net.decoder.layers[0].linear1 = moe_linear
        # moe_linear = LoRA(self.policy_head.ema_noise_pred_net.decoder.layers[0].linear2, dim=self.policy_head.ema_noise_pred_net.decoder.layers[0].linear2.in_features, rank=self.lora_rank,
        #                   dim_out=self.policy_head.ema_noise_pred_net.decoder.layers[0].linear2.out_features)
        # self.policy_head.ema_noise_pred_net.decoder.layers[0].linear2 = moe_linear
        # moe_linear = LoRA(self.policy_head.ema_noise_pred_net.decoder.layers[1].linear1, dim=self.policy_head.ema_noise_pred_net.decoder.layers[1].linear1.in_features, rank=self.lora_rank,
        #                   dim_out=self.policy_head.ema_noise_pred_net.decoder.layers[1].linear1.out_features)
        # self.policy_head.ema_noise_pred_net.decoder.layers[1].linear1 = moe_linear
        # moe_linear = LoRA(self.policy_head.ema_noise_pred_net.decoder.layers[1].linear2, dim=self.policy_head.ema_noise_pred_net.decoder.layers[1].linear2.in_features, rank=self.lora_rank,
        #                   dim_out=self.policy_head.ema_noise_pred_net.decoder.layers[1].linear2.out_features)
        # self.policy_head.ema_noise_pred_net.decoder.layers[1].linear2 = moe_linear
        # moe_linear = LoRA(self.policy_head.ema_noise_pred_net.head, dim=self.policy_head.ema_noise_pred_net.head.in_features, rank=self.lora_rank, dim_out=self.policy_head.ema_noise_pred_net.head.out_features)
        # self.policy_head.ema_noise_pred_net.head = moe_linear

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

    def calc(self, data):
        x = self.spatial_encode(data)  # (B, T, num_modality, E)
        x = self.temporal_encode(x)  # (B, T, E)
        return x

    def forward(self, data):
        x = self.spatial_encode(data)  # (B, T, num_modality, E)
        x = self.temporal_encode(x)  # (B, T, E)
        # action = self.policy_head(x).reshape(x.shape[0], self.num_queries, -1)  # (bs,1,70) -> (bs,10,7)
        action = self.policy_head(obs_seq=x, action_seq=data["actions"])  # (bs,1,70) -> (bs,10,7)
        return action
        # dist = self.policy_head(x)
        # return dist

    def get_action(self, data):
        self.eval()
        with torch.no_grad():
            with amp.autocast('cuda', dtype=torch.float16):
                data = self.preprocess_input(data, train_mode=False)
                x = self.spatial_encode(data)
                # self.latent_queue.append(x)
                # if len(self.latent_queue) > self.max_seq_len:
                #     self.latent_queue.pop(0)
                # x = torch.cat(self.latent_queue, dim=1)  # (B, T, H_all)
                x = self.temporal_encode(x)
                # dist = self.policy_head(x[:, -1])
                # action = self.policy_head(obs_seq=x, action_seq=None)  # (bs, 10, 7)
                action = self.policy_head(x).reshape(x.shape[0], self.num_queries, -1).to(torch.float32)
        # action = dist.sample().detach().cpu()
        # return action.view(action.shape[0], -1).numpy()
        bs = action.shape[0]
        actions = []
        for i in range(bs):
            self.all_time_actions[i, [self.step], self.step: self.step + self.num_queries] = action[[i]]
            actions_for_curr_step = self.all_time_actions[i, :, self.step]
            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            k = 0.01
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
            action_chunk = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)  # (bs, 7)
            actions.append(action_chunk)
        actions = torch.cat(actions, dim=0)
        self.step += 1
        return actions.detach().cpu().numpy()  # (bs, 7)

    def reset(self):
        self.latent_queue = []
        self.step = 0
        self.all_time_actions.zero_()

    def compute_loss(self, data, reduction="mean"):
        data = self.preprocess_input(data, train_mode=True)
        # action = self.forward(data)
        # loss = F.l1_loss(action, data["actions"], reduction=reduction)

        output = self.forward(data)
        noise_pred = output['noise_pred']
        noise = output['noise']
        loss = F.mse_loss(noise_pred, noise, reduction=reduction)
        return loss