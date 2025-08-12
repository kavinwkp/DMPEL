import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from torch import amp
import copy
import numpy as np

def generate_orthogonal_tensor(tensor, normalize=True):
    pool_size, in_dim, out_dim = tensor.shape
    M = tensor.reshape(pool_size, -1).T
    
    Q, _ = torch.linalg.qr(M, mode='reduced')
    
    rand_vec = torch.randn(in_dim*out_dim, device=tensor.device)
    proj = Q @ (Q.T @ rand_vec)
    ortho_vec = rand_vec - proj

    if torch.norm(ortho_vec) < 1e-12:
        ortho_vec = torch.linalg.svd(M).U[:,-1]
    
    new_matrix = ortho_vec.reshape(in_dim, out_dim)
    
    return new_matrix / torch.norm(new_matrix) if normalize else new_matrix


class LoRA(nn.Module):
    def __init__(
        self,
        projection: nn.Module,
        dim: int,
        rank: int = 16,
        alpha: int = 16, # scale = alpha / rank
        # dim_out: int = 0
    ):
        super().__init__()
        self.weight = projection.weight
        self.bias = projection.bias
        self.dim = dim
        self.rank = rank
        self.scale = alpha / rank
        self.A = nn.Linear(dim, rank, bias=False)
        # if dim_out != 0:
        #     self.B = nn.Linear(rank, dim_out, bias=False)
        # else:
        self.B = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.lora_attached = True

    def forward(self, x):
        orig = F.linear(x, self.weight, self.bias)  # Shape: (B, N, org_C)
        
        if self.lora_attached:
            new = self.B(self.A(x))
            orig += (self.scale * new)
        
        return orig
    
    def set_lora_attached(self, lora_attached: bool):
        self.lora_attached = lora_attached


class LoRAqkv(nn.Module):
    def __init__(
        self,
        qkv: nn.Module,
        dim: int,
        rank: int = 16,
        alpha: int = 16, # scale = lora_alpha / rank
    ):
        super().__init__()
        self.weight = qkv.weight
        self.bias = qkv.bias
        self.dim = dim
        self.rank = rank
        self.scale = alpha / rank
        self.A_q = nn.Linear(dim, rank, bias=False)
        self.B_q = nn.Linear(rank, dim, bias=False)
        self.A_v = nn.Linear(dim, rank, bias=False)
        self.B_v = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.A_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B_q.weight)
        nn.init.zeros_(self.B_v.weight)
        self.lora_attached = True

    def forward(self, x) -> torch.Tensor:
        # Compute the original qkv
        qkv = F.linear(x, self.weight, self.bias)  # Shape: (B, N, 3 * org_C)

        if self.lora_attached:
            # Compute the new q and v components
            new_q = self.B_q(self.A_q(x))
            new_v = self.B_v(self.A_v(x))

            # Add new q and v components to the original qkv tensor
            qkv[:, :, : self.dim] += (self.scale * new_q)
            qkv[:, :, -self.dim :] += (self.scale * new_v)

        return qkv

    def set_lora_attached(self, lora_attached: bool):
        self.lora_attached = lora_attached


class L2MLinear(nn.Module):
    def __init__(
        self,
        projection: nn.Module,
        pool_size: int,
    ):
        super().__init__()
        self.weight = projection.weight
        self.bias = projection.bias
        self.in_dim = projection.in_features
        self.out_dim = projection.out_features
        self.pool_size = pool_size
        self.W_pool = nn.Parameter(torch.zeros((self.pool_size, self.in_dim, self.out_dim)))
        self.B_pool = nn.Parameter(torch.zeros((self.pool_size, self.out_dim)))
        nn.init.zeros_(self.W_pool)
        nn.init.zeros_(self.B_pool)
        self.idx = None
        self.register_buffer('_frozen_mask', torch.zeros(pool_size, dtype=torch.bool))
        
    def forward(self, x) -> torch.Tensor:

        assert self.idx is not None
        idx = self.idx.squeeze(1)
        bsz = self.idx.shape[0]
        if bsz < x.shape[0] and x.dim() == 2:
            reshaped = True
            shape = (x.shape[0], self.out_dim)
            x = torch.reshape(x, (bsz, -1, self.in_dim))
        else:
            reshaped = False

        is_frozen = self._frozen_mask[idx]  # (bsz,)
        W = torch.where(
            is_frozen.view(-1, 1, 1),
            self.W_pool[idx].detach(),
            self.W_pool[idx]
        )
        B = torch.where(
            is_frozen.view(-1, 1),
            self.B_pool[idx].detach(),
            self.B_pool[idx]
        )
        if x.dim() == 3:
            out = torch.einsum('bni,bij->bnj', x, W) + B.unsqueeze(1)
        elif x.dim() == 2:
            out = torch.einsum('bi,bij->bj', x, W) + B
        else:
            raise NotImplementedError(f"Unsupported input dimension: {x.dim()}")

        if reshaped:
            out = torch.reshape(out, shape)
        
        return out
    
    def set_idx(self, idx):
        self.idx = idx

    def freeze_expert_in_pool(self, indices):
        if isinstance(indices, int):
            indices = [indices]
        self._frozen_mask[torch.tensor(indices)] = True

    def unfreeze_expert_in_pool(self, indices):
        if isinstance(indices, int):
            indices = [indices]
        self._frozen_mask[torch.tensor(indices)] = False

    @property
    def frozen_indices(self):
        return torch.where(self._frozen_mask)[0].tolist()


class L2MLoRA(nn.Module):
    def __init__(
        self,
        projection: nn.Module,
        pool_size: int,
        dim: int,
        rank: int = 8,
        alpha: int = 8, # scale = lora_alpha / rank
    ):
        super().__init__()
        self.weight = projection.weight
        self.bias = projection.bias
        self.dim = dim
        self.pool_size = pool_size
        self.rank = rank
        self.scale = alpha / rank
        self.A_pool = nn.Parameter(torch.zeros((self.pool_size, self.dim, self.rank)))
        self.B_pool = nn.Parameter(torch.zeros((self.pool_size, self.rank, self.dim)))
        nn.init.kaiming_uniform_(self.A_pool, a=math.sqrt(5))
        nn.init.zeros_(self.B_pool)
        self.idx = None
        self.register_buffer('_frozen_mask', torch.zeros(pool_size, dtype=torch.bool))
        
    
    def forward(self, x) -> torch.Tensor:
        # assert x.shape[0] == lora.shape[0] # B
        # Compute the original qkv
        orig = F.linear(x, self.weight, self.bias)  # Shape: (B, N, org_C)

        if self.idx is not None:
            idx = self.idx.squeeze(1)
            bsz = self.idx.shape[0]
            if (x.dim() == 3 and bsz < x.shape[0]):
                reshaped = True
                shape = orig.shape
                x = torch.reshape(x, (bsz, -1, self.dim))
                orig = torch.reshape(orig, (bsz, -1, self.dim))
            else:
                reshaped = False

            is_frozen = self._frozen_mask[idx]  # (bsz,)
            
            A = torch.where(
                is_frozen.view(-1, 1, 1),
                self.A_pool[idx].detach(),
                self.A_pool[idx]
            )
            B = torch.where(
                is_frozen.view(-1, 1, 1),
                self.B_pool[idx].detach(),
                self.B_pool[idx]
            )
            
            new = torch.einsum('bni,bir,brj->bnj', x, A, B)
            orig += (self.scale * new)

            if reshaped:
                orig = torch.reshape(orig, shape)
        
        return orig
    
    def set_idx(self, idx):
        self.idx = idx

    def freeze_expert_in_pool(self, indices):
        if isinstance(indices, int):
            indices = [indices]
        self._frozen_mask[torch.tensor(indices)] = True

    def unfreeze_expert_in_pool(self, indices):
        if isinstance(indices, int):
            indices = [indices]
        self._frozen_mask[torch.tensor(indices)] = False

    @property
    def frozen_indices(self):
        return torch.where(self._frozen_mask)[0].tolist()


class L2MLoRAqkv(nn.Module):
    def __init__(
        self,
        qkv: nn.Module,
        pool_size: int,
        dim: int,
        rank: int = 8,
        alpha: int = 8, # scale = lora_alpha / rank
    ):
        super().__init__()
        self.weight = qkv.weight
        self.bias = qkv.bias
        self.pool_size = pool_size
        self.dim = dim
        self.rank = rank
        self.scale = alpha / rank
        self.A_q_pool = nn.Parameter(torch.zeros((self.pool_size, self.dim, self.rank)))
        self.B_q_pool = nn.Parameter(torch.zeros((self.pool_size, self.rank, self.dim)))
        self.A_v_pool = nn.Parameter(torch.zeros((self.pool_size, self.dim, self.rank)))
        self.B_v_pool = nn.Parameter(torch.zeros((self.pool_size, self.rank, self.dim)))
        nn.init.kaiming_uniform_(self.A_q_pool, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.A_v_pool, a=math.sqrt(5))
        nn.init.zeros_(self.B_q_pool)
        nn.init.zeros_(self.B_v_pool)
        self.idx = None
        self.register_buffer('_frozen_mask', torch.zeros(pool_size, dtype=torch.bool))
        

    def forward(self, x) -> torch.Tensor:
        # idx: (B, 1)
        # Compute the original qkv
        qkv = F.linear(x, self.weight, self.bias)  # Shape: (B, N, 3 * org_C)

        if self.idx is not None:
            # Compute the new q and v components
            idx = self.idx.squeeze(1)
            
            bsz = self.idx.shape[0]
            if (x.dim() == 3 and bsz < x.shape[0]):
                reshaped = True
                shape = qkv.shape
                x = torch.reshape(x, (bsz, -1, self.dim))
                qkv = torch.reshape(qkv, (bsz, -1, 3*self.dim))
            else:
                reshaped = False

            
            is_frozen = self._frozen_mask[idx]  # (bsz,)
            
            A_q = torch.where(
                is_frozen.view(-1, 1, 1),
                self.A_q_pool[idx].detach(),
                self.A_q_pool[idx]
            )
            B_q = torch.where(
                is_frozen.view(-1, 1, 1),
                self.B_q_pool[idx].detach(),
                self.B_q_pool[idx]
            )
            A_v = torch.where(
                is_frozen.view(-1, 1, 1),
                self.A_v_pool[idx].detach(),
                self.A_v_pool[idx]
            )
            B_v = torch.where(
                is_frozen.view(-1, 1, 1),
                self.B_v_pool[idx].detach(),
                self.B_v_pool[idx]
            )
            
            new_q = torch.einsum('bni,bir,brj->bnj', x, A_q, B_q)
            new_v = torch.einsum('bni,bir,brj->bnj', x, A_v, B_v)

            # Add new q and v components to the original qkv tensor
            qkv[:, :, : self.dim] += (self.scale * new_q)
            qkv[:, :, -self.dim :] += (self.scale * new_v)

            if reshaped:
                qkv = torch.reshape(qkv, shape)

        return qkv

    def set_idx(self, idx):
        self.idx = idx

    def freeze_expert_in_pool(self, indices):
        if isinstance(indices, int):
            indices = [indices]
        self._frozen_mask[torch.tensor(indices)] = True

    def unfreeze_expert_in_pool(self, indices):
        if isinstance(indices, int):
            indices = [indices]
        self._frozen_mask[torch.tensor(indices)] = False

    @property
    def frozen_indices(self):
        return torch.where(self._frozen_mask)[0].tolist()


class MoELoRA(nn.Module):
    def __init__(
        self,
        projection: nn.Module,
        rank: int = 16,
        alpha: int = 16, # scale = lora_alpha / rank
        init_pool_size: int = 1,
        merge_AB: str='output',
        tune_bias: bool=True,
        lora_dropout: float=0.15,
    ):
        super().__init__()
        self.weight = projection.weight
        self.bias = projection.bias
        # self.weight.requires_grad = False
        # self.bias.requires_grad = False
        self.in_dim = projection.in_features
        self.out_dim = projection.out_features
        self.rank = rank
        self.scale = alpha / rank
        self.pool_size = init_pool_size
        self.dropout_ratio = lora_dropout
        self.merge_AB = merge_AB
        self.tune_bias = tune_bias
        
        self.A_pool = nn.Parameter(torch.zeros((self.pool_size, self.in_dim, self.rank)))
        self.B_pool = nn.Parameter(torch.zeros((self.pool_size, self.rank, self.out_dim)))
        nn.init.kaiming_uniform_(self.A_pool, a=math.sqrt(5))
        nn.init.zeros_(self.B_pool)
        if self.bias is not None and self.tune_bias:
            self.bias_pool = nn.Parameter(torch.zeros((self.pool_size, self.out_dim)))
            nn.init.zeros_(self.bias_pool)

        self.register_buffer('_frozen_mask', torch.zeros(self.pool_size, dtype=torch.bool))
        self.idx = None
        self.attn = None
        self.use_lora = True
    
    def set_attentions(self, topk_attn, topk_idx):
        self.attn = topk_attn
        self.idx = topk_idx
        if self.merge_AB == 'weight':

            bsz, k = self.idx.shape
            # (bsz, k)
            is_frozen = self._frozen_mask[self.idx].view(bsz, k, 1, 1)
            
            # (bsz, k, in_dim, rank)
            A = torch.where(
                is_frozen,
                self.A_pool[self.idx].detach(),
                self.A_pool[self.idx]
            )

            # (bsz, k, rank, out_dim)
            B = torch.where(
                is_frozen,
                self.B_pool[self.idx].detach(),
                self.B_pool[self.idx]
            )

            if self.dropout_ratio > 0:
                attn = F.dropout(self.attn, p=self.dropout_ratio, training=self.training)

            self.weighted_A = torch.einsum('bkir,bk->bir', A, attn)
            self.weighted_B = torch.einsum('bkro,bk->bro', B, attn)

            if self.bias is not None and self.tune_bias:
                is_frozen = self._frozen_mask[self.idx].view(bsz, k, 1)
                b = torch.where(
                        is_frozen,
                        self.bias_pool[self.idx].detach(),
                        self.bias_pool[self.idx]
                    )
                self.weighted_bias = torch.einsum('bko,bk->bo', b, attn)
    
    def set_use_lora(self, use_lora):
        self.use_lora = use_lora
    
    def freeze_expert_in_pool(self, indices):
        if isinstance(indices, int):
            indices = [indices]
        self._frozen_mask[torch.tensor(indices)] = True

    def unfreeze_expert_in_pool(self, indices):
        if isinstance(indices, int):
            indices = [indices]
        self._frozen_mask[torch.tensor(indices)] = False

    @property
    def frozen_indices(self):
        return torch.where(self._frozen_mask)[0].tolist()
    
    def forward(self, x) -> torch.Tensor:
        # Compute the original
        orig = F.linear(x, self.weight, self.bias)  # Shape: (B, T, org_C)
        
        if self.use_lora:
            assert self.idx is not None and self.attn is not None
            bsz, k = self.idx.shape
            assert k <= self.pool_size

            if x.dim() == 3 and bsz < x.shape[0]:
                reshaped = True
                shape = orig.shape
                x = torch.reshape(x, (bsz, -1, self.in_dim))
                orig = torch.reshape(orig, (bsz, -1, self.out_dim))
            elif x.dim() == 2:
                reshaped = True
                shape = (x.shape[0], self.out_dim)
                x = torch.reshape(x, (bsz, -1, self.in_dim))
                orig = torch.reshape(orig, (bsz, -1, self.out_dim))
            else:
                reshaped = False

            assert x.shape[0] == orig.shape[0] == bsz

            if self.merge_AB == 'output':
                # (bsz, k)
                is_frozen = self._frozen_mask[self.idx].view(bsz, k, 1, 1)
                
                # (bsz, k, in_dim, rank)
                A = torch.where(
                    is_frozen,
                    self.A_pool[self.idx].detach(),
                    self.A_pool[self.idx]
                )

                # (bsz, k, rank, out_dim)
                B = torch.where(
                    is_frozen,
                    self.B_pool[self.idx].detach(),
                    self.B_pool[self.idx]
                )
                
                mask = self.attn == 0 # (bsz, k)
                A.masked_fill_(mask.unsqueeze(-1).unsqueeze(-1).expand_as(A), 0.0)
                B.masked_fill_(mask.unsqueeze(-1).unsqueeze(-1).expand_as(B), 0.0)

                if self.bias is not None and self.tune_bias:
                    is_frozen = self._frozen_mask[self.idx].view(bsz, k, 1)
                    b = torch.where(
                        is_frozen,
                        self.bias_pool[self.idx].detach(),
                        self.bias_pool[self.idx]
                    )

            if self.dropout_ratio > 0:
                attn = F.dropout(self.attn, p=self.dropout_ratio, training=self.training)
            
            if self.merge_AB == 'output':
                outputs = torch.einsum('bni,bkir,bkro->bnko', x, A, B)
                weighted_output = torch.einsum('bnko,bk->bno', outputs, attn)
            elif self.merge_AB == 'weight':
                weighted_output = torch.einsum('bni,bir,brj->bnj', x, self.weighted_A, self.weighted_B)
            else:
                raise NotImplementedError

            orig += (self.scale * weighted_output)

            if self.bias is not None and self.tune_bias:
                if self.merge_AB == 'weight':
                    orig += (self.scale * self.weighted_bias.unsqueeze(1))
                else:
                    bias = torch.einsum('bko,bk->bo', b, attn).unsqueeze(1)
                    orig += (self.scale * bias)

            if reshaped:
                orig = torch.reshape(orig, shape)
        
        return orig
    
    def add_expert(self, num_experts: int = 1, init_method: str='random'):
        if self.pool_size > 0:
            self.freeze_expert_in_pool(range(self.pool_size))
        if num_experts > 0:
            device = self.A_pool.device
            if init_method == 'random' or self.pool_size == 0:
                new_A = torch.zeros((num_experts, self.in_dim, self.rank), device=device)
                new_B = torch.zeros((num_experts, self.rank, self.out_dim), device=device)

                nn.init.kaiming_uniform_(new_A, a=math.sqrt(5))
                nn.init.zeros_(new_B)
                
                self.A_pool = nn.Parameter(torch.cat([self.A_pool.data, new_A], dim=0), requires_grad=True)
                self.B_pool = nn.Parameter(torch.cat([self.B_pool.data, new_B], dim=0), requires_grad=True)
            
            elif init_method == 'ortho':
                for i in range(num_experts):
                    new_A = generate_orthogonal_tensor(self.A_pool.data)
                    self.A_pool = nn.Parameter(torch.cat([self.A_pool.data, new_A.unsqueeze(0)], dim=0), requires_grad=True)
                new_B = torch.zeros((num_experts, self.rank, self.out_dim), device=device)
                nn.init.zeros_(new_B)
                self.B_pool = nn.Parameter(torch.cat([self.B_pool.data, new_B], dim=0), requires_grad=True)
            else:
                raise NotImplementedError
            
            if self.bias is not None and self.tune_bias:
                new_bias = torch.zeros((num_experts, self.out_dim), device=device)
                self.bias_pool = nn.Parameter(torch.cat([self.bias_pool.data, new_bias], dim=0), requires_grad=True)
            
            new_frozen = torch.zeros(num_experts, dtype=torch.bool, device=device)
            self._frozen_mask = torch.cat([self._frozen_mask, new_frozen])
            
            self.pool_size += num_experts


class MoELoRAqkv(nn.Module):
    def __init__(
        self,
        qkv: nn.Module,
        rank: int = 16,
        alpha: int = 16, # scale = lora_alpha / rank
        init_pool_size: int = 1,
        merge_AB: str='output',
        tune_bias: bool=True,
        lora_dropout: float=0.15,
    ):
        super().__init__()
        self.weight = qkv.weight
        self.bias = qkv.bias
        # self.weight.requires_grad = False
        # self.bias.requires_grad = False
        self.dim = qkv.in_features
        self.rank = rank
        self.scale = alpha / rank
        self.pool_size = init_pool_size
        self.dropout_ratio = lora_dropout
        self.merge_AB = merge_AB
        self.tune_bias = tune_bias
        
        self.A_q_pool = nn.Parameter(torch.zeros((self.pool_size, self.dim, self.rank)))
        self.B_q_pool = nn.Parameter(torch.zeros((self.pool_size, self.rank, self.dim)))
        self.A_v_pool = nn.Parameter(torch.zeros((self.pool_size, self.dim, self.rank)))
        self.B_v_pool = nn.Parameter(torch.zeros((self.pool_size, self.rank, self.dim)))
        nn.init.kaiming_uniform_(self.A_q_pool, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.A_v_pool, a=math.sqrt(5))
        nn.init.zeros_(self.B_q_pool)
        nn.init.zeros_(self.B_v_pool)

        if self.bias is not None and self.tune_bias:
            self.bias_pool = nn.Parameter(torch.zeros((self.pool_size, 3 * self.dim)))
            nn.init.zeros_(self.bias_pool)

        self.register_buffer('_frozen_mask', torch.zeros(self.pool_size, dtype=torch.bool))
        self.idx = None
        self.attn = None
        self.use_lora = False
        
    def set_attentions(self, topk_attn, topk_idx):
        self.attn = topk_attn
        self.idx = topk_idx

        if self.merge_AB == 'weight':

            bsz, k = self.idx.shape

            if self.dropout_ratio > 0:
                attn = F.dropout(self.attn, p=self.dropout_ratio, training=self.training)
            
            # (bsz, k)
            is_frozen = self._frozen_mask[self.idx].view(bsz, k, 1, 1)

            # (bsz, k, in_dim, rank)
            A_q = torch.where(
                is_frozen,
                self.A_q_pool[self.idx].detach(),
                self.A_q_pool[self.idx]
            )
            A_v = torch.where(
                is_frozen,
                self.A_v_pool[self.idx].detach(),
                self.A_v_pool[self.idx]
            )

            # (bsz, k, rank, out_dim)
            B_q = torch.where(
                is_frozen,
                self.B_q_pool[self.idx].detach(),
                self.B_q_pool[self.idx]
            )
            B_v = torch.where(
                is_frozen,
                self.B_v_pool[self.idx].detach(),
                self.B_v_pool[self.idx]
            )
            self.weighted_A_q = torch.einsum('bkir,bk->bir', A_q, attn)
            self.weighted_B_q = torch.einsum('bkro,bk->bro', B_q, attn)
            self.weighted_A_v = torch.einsum('bkir,bk->bir', A_v, attn)
            self.weighted_B_v = torch.einsum('bkro,bk->bro', B_v, attn)

            is_frozen = self._frozen_mask[self.idx].view(bsz, k, 1)
            if self.bias is not None and self.tune_bias:
                b = torch.where(
                    is_frozen,
                    self.bias_pool[self.idx].detach(),
                    self.bias_pool[self.idx]
                )
                self.weighted_bias = torch.einsum('bko,bk->bo', b, attn).unsqueeze(1)
    
    def set_use_lora(self, use_lora):
        self.use_lora = use_lora
    
    def freeze_expert_in_pool(self, indices):
        if isinstance(indices, int):
            indices = [indices]
        self._frozen_mask[torch.tensor(indices)] = True

    def unfreeze_expert_in_pool(self, indices):
        if isinstance(indices, int):
            indices = [indices]
        self._frozen_mask[torch.tensor(indices)] = False

    @property
    def frozen_indices(self):
        return torch.where(self._frozen_mask)[0].tolist()
    
    def forward(self, x) -> torch.Tensor:
        # Compute the original qkv
        qkv = F.linear(x, self.weight, self.bias)  # Shape: (B, N, 3 * org_C)
        
        if self.use_lora:
            assert self.idx is not None and self.attn is not None
            bsz, k = self.idx.shape
            assert k <= self.pool_size

            if x.dim() == 3 and bsz < x.shape[0]:
                reshaped = True
                shape = qkv.shape
                x = torch.reshape(x, (bsz, -1, self.dim))
                qkv = torch.reshape(qkv, (bsz, -1, 3*self.dim))
            else:
                reshaped = False
                

            assert x.dim() == 3 and x.shape[0] == qkv.shape[0] == bsz

            if self.merge_AB == 'output':
                # (bsz, k)
                is_frozen = self._frozen_mask[self.idx].view(bsz, k, 1, 1)

                # (bsz, k, in_dim, rank)
                A_q = torch.where(
                    is_frozen,
                    self.A_q_pool[self.idx].detach(),
                    self.A_q_pool[self.idx]
                )
                A_v = torch.where(
                    is_frozen,
                    self.A_v_pool[self.idx].detach(),
                    self.A_v_pool[self.idx]
                )

                if self.fix_A:
                    A_q = A_q.detach()
                    A_v = A_v.detach()

                # (bsz, k, rank, out_dim)
                B_q = torch.where(
                    is_frozen,
                    self.B_q_pool[self.idx].detach(),
                    self.B_q_pool[self.idx]
                )
                B_v = torch.where(
                    is_frozen,
                    self.B_v_pool[self.idx].detach(),
                    self.B_v_pool[self.idx]
                )

                mask = self.attn == 0 # (bsz, k)
                A_q.masked_fill_(mask.unsqueeze(-1).unsqueeze(-1).expand_as(A_q), 0.0)
                A_v.masked_fill_(mask.unsqueeze(-1).unsqueeze(-1).expand_as(A_v), 0.0)
                B_q.masked_fill_(mask.unsqueeze(-1).unsqueeze(-1).expand_as(B_q), 0.0)
                B_v.masked_fill_(mask.unsqueeze(-1).unsqueeze(-1).expand_as(B_v), 0.0)
                
                if self.bias is not None and self.tune_bias:
                    is_frozen = self._frozen_mask[self.idx].view(bsz, k, 1)
                    b = torch.where(
                        is_frozen,
                        self.bias_pool[self.idx].detach(),
                        self.bias_pool[self.idx]
                    )
            
            if self.dropout_ratio > 0:
                attn = F.dropout(self.attn, p=self.dropout_ratio, training=self.training)
            
            if self.merge_AB == 'output':
                new_q = torch.einsum('bni,bkir,bkro->bnko', x, A_q, B_q)
                new_v = torch.einsum('bni,bkir,bkro->bnko', x, A_v, B_v)
                weighted_new_q = torch.einsum('bnko,bk->bno', new_q, attn)
                weighted_new_v = torch.einsum('bnko,bk->bno', new_v, attn)
            elif self.merge_AB == 'weight':
                weighted_new_q = torch.einsum('bni,bir,brj->bnj', x, self.weighted_A_q, self.weighted_B_q)
                weighted_new_v = torch.einsum('bni,bir,brj->bnj', x, self.weighted_A_v, self.weighted_B_v)
            else:
                raise NotImplementedError
            
            qkv[..., :self.dim] += (self.scale * weighted_new_q)
            qkv[..., -self.dim:] += (self.scale * weighted_new_v)
            
            if self.bias is not None and self.tune_bias:
                if self.merge_AB == 'weight':
                    qkv += (self.scale * self.weighted_bias)
                elif self.merge_AB == 'output':
                    bias = torch.einsum('bko,bk->bo', b, attn).unsqueeze(1)
                    qkv += (self.scale * bias)
                else:
                    raise NotImplementedError
            
            if reshaped:
                qkv = torch.reshape(qkv, shape)
            
        return qkv

    def add_expert(self, num_experts: int = 1, init_method: str='random'):
        if self.pool_size > 0:
            self.freeze_expert_in_pool(range(self.pool_size))
        
        if num_experts > 0:
            device = self.A_q_pool.device

            if init_method == 'random' or self.pool_size == 0:
                new_A_q = torch.zeros((num_experts, self.dim, self.rank), device=device)
                new_B_q = torch.zeros((num_experts, self.rank, self.dim), device=device)
                new_A_v = torch.zeros((num_experts, self.dim, self.rank), device=device)
                new_B_v = torch.zeros((num_experts, self.rank, self.dim), device=device)
                
                nn.init.kaiming_uniform_(new_A_q, a=math.sqrt(5))
                nn.init.kaiming_uniform_(new_A_v, a=math.sqrt(5))
                nn.init.zeros_(new_B_q)
                nn.init.zeros_(new_B_v)

                self.A_q_pool = nn.Parameter(torch.cat([self.A_q_pool.data, new_A_q], dim=0), requires_grad=True)
                self.B_q_pool = nn.Parameter(torch.cat([self.B_q_pool.data, new_B_q], dim=0), requires_grad=True)
                self.A_v_pool = nn.Parameter(torch.cat([self.A_v_pool.data, new_A_v], dim=0), requires_grad=True)
                self.B_v_pool = nn.Parameter(torch.cat([self.B_v_pool.data, new_B_v], dim=0), requires_grad=True)
            
            elif init_method == 'ortho':
                for i in range(num_experts):
                    new_A_q = generate_orthogonal_tensor(self.A_q_pool.data)
                    new_A_v = generate_orthogonal_tensor(self.A_v_pool.data)
                    self.A_q_pool = nn.Parameter(torch.cat([self.A_q_pool.data, new_A_q.unsqueeze(0)], dim=0), requires_grad=True)
                    self.A_v_pool = nn.Parameter(torch.cat([self.A_v_pool.data, new_A_v.unsqueeze(0)], dim=0), requires_grad=True)
                
                new_B_q = torch.zeros((num_experts, self.rank, self.dim), device=device)
                new_B_v = torch.zeros((num_experts, self.rank, self.dim), device=device)
                nn.init.zeros_(new_B_q)
                nn.init.zeros_(new_B_v)
                self.B_q_pool = nn.Parameter(torch.cat([self.B_q_pool.data, new_B_q], dim=0), requires_grad=True)
                self.B_v_pool = nn.Parameter(torch.cat([self.B_v_pool.data, new_B_v], dim=0), requires_grad=True)
            
            else:
                raise NotImplementedError
            
            if self.bias is not None and self.tune_bias:
                new_bias = torch.zeros((num_experts, 3 * self.dim), device=device)
                self.bias_pool = nn.Parameter(torch.cat([self.bias_pool.data, new_bias], dim=0), requires_grad=True)

            new_frozen = torch.zeros(num_experts, dtype=torch.bool, device=device)
            self._frozen_mask = torch.cat([self._frozen_mask, new_frozen])
            
            self.pool_size += num_experts


class MoERouterCoeff(nn.Module):
    def __init__(self, in_dim, hidden_dim, cfg):
        super().__init__()
        self.in_dim = in_dim
        self.pool_size = cfg.init_pool_size
        self.hidden_dim = hidden_dim
        self.router_coeff_seperate = cfg.router_coeff_seperate
        if self.router_coeff_seperate:
            self.out_dim = self.pool_size * 6
        else:
            self.out_dim = self.pool_size
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                        nn.GELU('tanh'),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.GELU('tanh'),
                                        nn.Linear(hidden_dim, self.out_dim),
                                        nn.Sigmoid(),
                                        )
        self.last_task_pool_size = self.pool_size
        self.moe_topk = cfg.moe_topk
    
    def forward(self, query_in):
        topk = min(self.moe_topk, self.pool_size)
        bsz = query_in.shape[0]
        coeff = self.encoder(query_in)
        coeff = 2 * coeff
        if self.router_coeff_seperate:
            coeff_multi = coeff.view(bsz, 6, self.pool_size)
            topk_coeff_multi, topk_idx_multi = torch.topk(coeff_multi, k=topk, dim=-1, sorted=False)
            topk_coeff = topk_coeff_multi.view(bsz, -1)
            topk_idx = topk_idx_multi.view(bsz, -1)
        else:
            topk_coeff, topk_idx = torch.topk(coeff, topk, dim=-1, sorted=False)
        
        return coeff, topk_coeff, topk_idx
    
    def add_expert(self, num_experts: int = 1):
        device = self.encoder[0].weight.device
        if num_experts > 0:
            new_pool_size = self.pool_size + num_experts
            if self.router_coeff_seperate:
                new_out_dim = new_pool_size * 6
            else:
                new_out_dim = new_pool_size
            new_encoder = nn.Sequential(nn.Linear(self.in_dim, self.hidden_dim),
                                        nn.GELU('tanh'),
                                        nn.Linear(self.hidden_dim, self.hidden_dim),
                                        nn.GELU('tanh'),
                                        nn.Linear(self.hidden_dim, new_out_dim),
                                        nn.Sigmoid(),
                                        )
            
            new_topk = min(self.moe_topk, new_pool_size)
            target_output = 0.5 / new_topk
            pre_act = np.log(target_output / (1 - target_output))
            nn.init.constant_(new_encoder[-2].bias, pre_act)
        
        self.encoder = new_encoder.to(device)
        self.last_task_pool_size = self.pool_size
        self.pool_size = new_pool_size
        self.out_dim = new_out_dim