"""Vision Transformer backbone components used by detection models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, Mlp, trunc_normal_


def _to_2tuple(value: int | Tuple[int, int]) -> Tuple[int, int]:
    """Ensure patch / image sizes are represented as (H, W)."""

    if isinstance(value, tuple):
        return value
    return (value, value)


@dataclass
class ShapeSpec:
    """Basic shape specification about a tensor."""

    channels: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    stride: Optional[int] = None


@dataclass
class ViTBackboneConfig:
    """Configuration for constructing a ViT backbone with optional FPN head."""

    img_size: int | Tuple[int, int] = 512
    patch_size: int | Tuple[int, int] = 16
    in_chans: int = 3
    embed_dim: int = 384
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    norm_layer: Type[nn.Module] = nn.LayerNorm
    act_layer: Type[nn.Module] = nn.GELU
    window_size: int = 14
    window_block_indexes: Tuple[int, ...] = (0, 1, 3, 4, 6, 7, 9, 10)
    drop_path_rate: float = 0.0
    pretrain_img_size: int | Tuple[int, int] = 224
    out_feature: str = "last_feat"
    pos_embed_init_std: float = 0.02
    freeze_patch_embed: bool = False
    frozen_stages: int = 0  # Number of leading Transformer blocks to freeze
    weights_path: Optional[str] = None

    # FPN attachment knobs
    out_channels: int = 256
    scale_factors: Tuple[float, ...] = (4.0, 2.0, 1.0, 0.5)
    square_pad: int = 512
    norm: str = "LN"
    use_last_level_max_pool: bool = True


class PatchEmbed(nn.Module):
    """Image to patch embedding using a conv projection."""

    def __init__(self, in_chans: int = 3, embed_dim: int = 768, patch_size: int | Tuple[int, int] = 16) -> None:
        super().__init__()
        patch_size = _to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ===== STUDENT TODO: Implement patch embedding =====
        # Hint: Convert image patches to embeddings:
        # 1. Use self.proj (conv2d) to project patches to embedding dimension
        # 2. Permute from (B, C, H, W) to (B, H, W, C) format for transformer
        # This is the crucial first step of ViT that converts images to tokens
        x = self.proj(x)

        x = x.permute(0, 2, 3, 1).contiguous()
        
        return x

        raise NotImplementedError("PatchEmbed.forward() not implemented")
        # ===================================================


def get_rel_pos(q_size, k_size, rel_pos):
    """Get relative positional embeddings according to query/key shapes."""

    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    """Add decomposed relative positional embeddings from mvitv2."""

    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(
        B, q_h * q_w, k_h * k_w
    )
    return attn


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        if input_size is None:
            raise ValueError("input_size must be provided for Attention to set relative positional parameters.")

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        input_size = _to_2tuple(input_size)
        self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ===== STUDENT TODO: Implement Multi-Head Attention =====
        # Hint: Follow the standard attention mechanism:
        # 1. Get input dimensions (B, H, W, C)
        # 2. Apply QKV linear transformation and reshape for multi-head
        # 3. Compute attention scores: Q @ K^T, scaled by sqrt(d_k)
        # 4. Add relative position embeddings using add_decomposed_rel_pos
        # 5. Apply softmax to get attention weights
        # 6. Apply attention to values: Attention @ V
        # 7. Reshape and apply output projection

        # b, h, w, c = x.shape
        # d_k = c // self.num_heads
        # n = h * w
        

        # qkv = self.qkv(x).reshape(b, h*w, 3, self.num_heads, d_k)

        # qkv = qkv.permute(2, 0, 3, 1, 4) 
        # q, k, v = qkv[0], qkv[1], qkv[2]

        # attn = (q @ k.transpose(-1, -2)) * self.scale

        # for h_i in range(self.num_heads):
        #     Q_head = q[:, h_i, :, :]  # shape (B, N, d_k)
        #     attn[:, h_i, :, :] += add_decomposed_rel_pos(
        #         attn[:, h_i, :, :], Q_head, self.rel_pos_h, self.rel_pos_w, (h_i, w), (h_i, w)
        # )

        # attn = attn.softmax(dim=-1)

        # out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        # out = self.proj(out)
        # out = out.view(b, h, w, c)

        # return out
        B, H, W, C = x.shape
        d_k = C // self.num_heads
        
        qkv = self.qkv(x).reshape(B, H*W, 3, self.num_heads, d_k)
        
        Q, K, V = qkv.unbind(2)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)


        attn_scores = (Q @ torch.transpose(K, -1, -2)) / math.sqrt(d_k)
        B, HN, N, _ = attn_scores.shape
        for h in range(self.num_heads):
            Q_head = Q[:, h, :, :]
            attn_scores[:, h, :, :] += add_decomposed_rel_pos(
                attn_scores[:, h, :, :], Q_head, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )
        
        attn_weigths = torch.softmax(attn_scores, dim=-1)
        out = (attn_weigths @ V).transpose(1,2).reshape(B,H,W,C)
        self.proj(out)
        return out
        raise NotImplementedError("Attention.forward() not implemented")
        # ==========================================================


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Partition into non-overlapping windows with padding if needed."""

    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(x: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]) -> torch.Tensor:
    """Reverse window partition and remove padding."""

    Hp, Wp = pad_hw
    H, W = hw
    B = x.shape[0] // (Hp * Wp // window_size // window_size)
    x = x.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class Block(nn.Module):
    """Transformer block with optional windowed attention and DropPath."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        attn_input = input_size if window_size == 0 else (window_size, window_size)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            input_size=attn_input,
        )

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ===== STUDENT TODO: Implement Transformer Block =====
        # Hint: Follow the standard Transformer block structure:
        # 1. Save input as shortcut for skip connection
        # 2. Apply layer norm and attention (handle windowed attention if needed)
        # 3. Add skip connection with dropout path: shortcut + drop_path(attention_output)
        # 4. Apply second layer norm and MLP
        # 5. Add second skip connection with dropout path
        # Remember to handle window partitioning when window_size > 0
        identity = x

        x = self.norm1(x)
        
        if self.window_size > 0:
            b, h, w, c = x.shape
            windows, (Hp, Wp) = window_partition(x, self.window_size)
            x = self.attn(windows)
            x = window_unpartition(x, self.window_size, (Hp, Wp), (h, w))
        else:
            x = self.attn(x)

        
        x = identity + self.drop_path(x)

        identity2 = x

        x = self.norm2(x)
        x = self.mlp(x)

        x = identity2 + self.drop_path(x)

        return x

        raise NotImplementedError("Block.forward() not implemented")
        # ==========================================================


def get_abs_pos(abs_pos: torch.Tensor, hw: Tuple[int, int], has_cls_token: bool = False) -> torch.Tensor:
    """Resize absolute positional embeddings to match input resolution."""
    # ===== STUDENT TODO: Implement positional encoding resize =====
    # Hint: Resize positional embeddings to match input resolution:
    # 1. Extract height and width from hw tuple
    # 2. Handle class token if present (remove from abs_pos)
    # 3. Determine original grid size from abs_pos.shape[1]
    # 4. If size mismatch, use F.interpolate to resize embeddings
    # 5. Return reshaped positional embeddings in (1, H, W, C) format
    # This allows ViT to handle different input sizes than pretraining
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1, :]

    num_patches = abs_pos.shape[1]
    orig_size = int(math.sqrt(num_patches))
    c = abs_pos.shape[2]

    abs_pos = abs_pos.reshape(1, orig_size, orig_size, c).permute(0, 3, 1, 2)

    if (orig_size != h) or (orig_size != w):
        abs_pos = F.interpolate(abs_pos, size=(h, w), mode="bilinear", align_corners=False)
    
    abs_pos = abs_pos.permute(0, 2, 3, 1).contiguous()

    return abs_pos

    raise NotImplementedError("get_abs_pos() not implemented")
    # ==============================================================


class ViT(nn.Module):
    """Vision Transformer backbone used in ViTDet."""

    def __init__(
        self,
        img_size: int | Tuple[int, int] = 512,
        patch_size: int | Tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        window_size: int = 0,
        window_block_indexes: Tuple[int, ...] = (),
        pretrain_img_size: int | Tuple[int, int] = 224,
        out_feature: str = "last_feat",
        drop_path_rate: float = 0.0,
        pos_embed_init_std: float = 0.02,
    ) -> None:
        super().__init__()

        self.img_size = _to_2tuple(img_size)
        self.patch_size = _to_2tuple(patch_size)
        self.pretrain_img_size = _to_2tuple(pretrain_img_size)

        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim, patch_size=self.patch_size)

        pretrain_grid = (
            self.pretrain_img_size[0] // self.patch_size[0],
            self.pretrain_img_size[1] // self.patch_size[1],
        )
        num_patches = pretrain_grid[0] * pretrain_grid[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        spatial_input = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                window_size=window_size if i in window_block_indexes else 0,
                input_size=spatial_input,
                drop_path=dpr[i],
            )
            self.blocks.append(block)

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: self.patch_size[0]}
        self._out_features = [out_feature]

        self.apply(self._init_weights)
        trunc_normal_(self.pos_embed, std=pos_embed_init_std)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, LayerNorm)):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # ===== STUDENT TODO: Implement ViT forward pass =====
        # Hint: Follow the Vision Transformer pipeline:
        # 1. Apply patch embedding to convert image to patches
        # 2. Get and add positional embeddings using get_abs_pos
        # 3. Pass through all transformer blocks sequentially
        # 4. Return output in the expected format (permute to BCHW)
        # Note: Output should be a dict with the feature name as key
        x = self.patch_embed(x)

        pos_embed = get_abs_pos(abs_pos=self.pos_embed, hw=(x.shape[1], x.shape[2]), has_cls_token=False)
        x = x + pos_embed

        for block in self.blocks:
            x = block(x)

        x = x.permute(0, 3, 1, 2).contiguous()

        return {self._out_features[0]: x}

        raise NotImplementedError("ViT.forward() not implemented")
        # ========================================================

    def output_shape(self) -> dict[str, ShapeSpec]:
        return {
            name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name])
            for name in self._out_features
        }


class Conv2d(torch.nn.Conv2d):
    """A wrapper around torch.nn.Conv2d to support additional features."""

    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class LayerNorm(nn.Module):
    """LayerNorm variant used in convolutional settings."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SimpleFeaturePyramid(nn.Module):
    """Implementation of the SimpleFeaturePyramid used in ViTDet."""

    def __init__(
        self,
        net: ViT,
        in_feature: str,
        out_channels: int,
        scale_factors: Tuple[float, ...],
        top_block: Optional[nn.Module] = None,
        norm: str = "LN",
        square_pad: int = 0,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.scale_factors = scale_factors

        input_shapes = net.output_shape()
        strides = [int(input_shapes[in_feature].stride / scale) for scale in scale_factors]

        dim = input_shapes[in_feature].channels
        self.stages: list[nn.Module] = []
        use_bias = norm == ""
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    LayerNorm(dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    Conv2d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                        bias=use_bias,
                        norm=LayerNorm(out_channels),
                    ),
                    Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=LayerNorm(out_channels),
                    ),
                ]
            )
            stage = nn.Sequential(*layers)
            level = int(math.log2(strides[idx]))
            self.add_module(f"simfp_{level}", stage)
            self.stages.append(stage)

        self.net = net
        self.in_feature = in_feature
        self.top_block = top_block
        self._out_feature_strides = {f"p{int(math.log2(s))}": s for s in strides}
        if self.top_block is not None:
            stage = int(math.log2(strides[-1]))
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides[f"p{s + 1}"] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # ===== STUDENT TODO: Implement ViT-FPN forward pass =====
        # Hint: Build feature pyramid from ViT backbone:
        # 1. Get bottom-up features from ViT backbone (self.net)
        # 2. Extract the main feature using self.in_feature key
        # 3. Apply each FPN stage to create multi-scale features
        # 4. Handle top_block if present for additional pyramid levels
        # 5. Return dictionary mapping feature names to tensors
        # This creates the multi-scale features needed for detection
        features = self.net(x)

        main_features = features[self.in_feature]

        output_features = {}

        for idx, stages in enumerate(self.stages):
            output_features[self._out_features[idx]] = stages(main_features)

        if self.top_block is not None:
            last_idx = len(self.stages)-1
            last_feature = output_features[self._out_features[last_idx]] 
            top_features = self.top_block(last_feature)

            for i, feature in enumerate(top_features):
                idx = len(self.stages) + i + 2
                output_features[f"p{idx}"] = feature
        else:
            # fallback: manually downsample p5 to create p6 if top_block is None
            output_features['p6'] = F.max_pool2d(output_features['p5'], kernel_size=1, stride=2)

        return output_features

        raise NotImplementedError("SimpleFeaturePyramid.forward() not implemented")
        # ============================================================


class LastLevelMaxPool(nn.Module):
    """Generate an extra P6 feature map by max pooling P5."""

    def __init__(self) -> None:
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


def _freeze_vit_layers(model: ViT, frozen_stages: int) -> None:
    if frozen_stages <= 0:
        return
    for idx, block in enumerate(model.blocks):
        if idx < frozen_stages:
            for param in block.parameters():
                param.requires_grad = False


def build_vit_backbone(config: Optional[ViTBackboneConfig] = None) -> ViT:
    """Instantiate a ViT backbone according to ``config``."""

    config = config or ViTBackboneConfig()
    vit = ViT(
        img_size=config.img_size,
        patch_size=config.patch_size,
        in_chans=config.in_chans,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        qkv_bias=config.qkv_bias,
        norm_layer=config.norm_layer,
        act_layer=config.act_layer,
        window_size=config.window_size,
        window_block_indexes=config.window_block_indexes,
        pretrain_img_size=config.pretrain_img_size,
        out_feature=config.out_feature,
        drop_path_rate=config.drop_path_rate,
        pos_embed_init_std=config.pos_embed_init_std,
    )

    if config.weights_path:
        state_dict = torch.load(config.weights_path, map_location="cpu")
        vit.load_state_dict(state_dict, strict=False)

    if config.freeze_patch_embed:
        for param in vit.patch_embed.parameters():
            param.requires_grad = False

    _freeze_vit_layers(vit, config.frozen_stages)
    return vit


def build_vit_fpn_backbone(config: Optional[ViTBackboneConfig] = None) -> SimpleFeaturePyramid:
    """Build a ViT backbone wrapped with the SimpleFeaturePyramid."""

    config = config or ViTBackboneConfig()
    vit = build_vit_backbone(config)
    top_block = LastLevelMaxPool() if config.use_last_level_max_pool else None
    return SimpleFeaturePyramid(
        net=vit,
        in_feature=config.out_feature,
        out_channels=config.out_channels,
        scale_factors=config.scale_factors,
        top_block=top_block,
        norm=config.norm,
        square_pad=config.square_pad,
    )


__all__ = [
    "ShapeSpec",
    "ViTBackboneConfig",
    "PatchEmbed",
    "Attention",
    "Block",
    "ViT",
    "SimpleFeaturePyramid",
    "LastLevelMaxPool",
    "build_vit_backbone",
    "build_vit_fpn_backbone",
]