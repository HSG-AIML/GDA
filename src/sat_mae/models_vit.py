# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm.models.vision_transformer
from timm.models.helpers import checkpoint_seq
from src.sat_mae.util.pos_embed import get_2d_sincos_pos_embed


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        # Added by Samar, need default pos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


### not used


class SharedAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.down = nn.Linear(in_dim, hidden_dim)
        self.up = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.up(self.down(x))


class ViTAdapted(nn.Module):
    def __init__(self, vit, adapter_hidden=8):
        super().__init__()
        self.vit = vit
        # qkv_out, qkv_in = self.vit.transformer.layers[2][0].to_qkv.weight.shape
        qkv_out, qkv_in = self.vit.blocks[0].attn.qkv.weight.shape
        self.adapter = SharedAdapter(qkv_in, qkv_out, adapter_hidden)

    def forward_attention(self, attention_module, x):
        B, N, C = x.shape
        qkv = attention_module.qkv(x)

        # add adapter vals
        qkv += self.adapter(x)

        qkv = qkv.reshape(
            B, N, 3, attention_module.num_heads, C // attention_module.num_heads
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * attention_module.scale
        attn = attn.softmax(dim=-1)
        attn = attention_module.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attention_module.proj(x)
        x = attention_module.proj_drop(x)
        return x

    def forward_block(self, x, block_module):
        x = x + block_module.drop_path1(
            block_module.ls1(
                self.forward_attention(block_module.attn, block_module.norm1(x))
            )
        )
        x = x + block_module.drop_path2(
            block_module.ls2(block_module.mlp(block_module.norm2(x)))
        )
        return x

    def forward_features(self, x):
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.norm_pre(x)
        if self.vit.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.vit.blocks:
                x = checkpoint_seq(self.forward_block(x, block))
        else:
            for block in self.vit.blocks:
                x = self.forward_block(x, block)
        x = self.vit.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.vit.forward_head(x)
        return x
