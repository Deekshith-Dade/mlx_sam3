from curses import window
from functools import partial
import math
from typing import Callable, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .model_misc import Mlp, LayerScale , DropPath

def polar(a, b):
    return (a * mx.exp(1j * b)).astype(mx.complex64)

def real(x: mx.array) -> mx.array:
    parts = mx.view(x, mx.float32).reshape(*x.shape, 2)
    return parts.reshape(*x.shape[:-1], -1)

def view_as_complex(x: mx.array) -> mx.array:
    assert x.shape[-1] % 2 == 0
    new_shape = list(x.shape[:-1]) + [-1, 2]
    parts = mx.reshape(x, new_shape)
    return (parts[..., 0] + 1j * parts[..., 1]).astype(mx.complex64)

def init_t_xy(
    end_x: int, end_y: int, scale: float = 1.0, offset: int = 0
) -> Tuple[mx.array, mx.array]:
    t = mx.arange(end_x * end_y, dtype=mx.float32)
    t_x = (t % end_x).astype(mx.float32)
    t_y = mx.divide(t, end_x).astype(mx.float32)
    return t_x * scale + offset, t_y * scale + offset

def compute_axial_cis(
    dim: int,
    end_x: int,
    end_y: int,
    theta: float = 10000.0,
    scale_pos: float = 1.0,
    offset: int = 0
) -> mx.array:
    freqs_x = 1.0 / (theta ** (mx.arange(0, dim, 4)[: (dim // 4)].astype(mx.float32) / dim))
    freqs_y = 1.0 / (theta ** (mx.arange(0, dim, 4)[: (dim // 4)].astype(mx.float32) / dim))

    t_x, t_y = init_t_xy(end_x, end_y, scale=scale_pos, offset=offset)
    freqs_x = mx.outer(t_x, freqs_x)
    freqs_y = mx.outer(t_y, freqs_y)
    freqs_cis_x = polar(mx.ones_like(freqs_x), freqs_x) 
    freqs_cis_y = polar(mx.ones_like(freqs_y), freqs_y)
    return mx.concat([freqs_cis_x, freqs_cis_y], axis=-1)

def reshape_for_broadcast(freqs_cis: mx.array, x: mx.array) -> mx.array:
    ndim = x.ndim

    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return mx.reshape(freqs_cis, shape)

def apply_rotary_enc(
    xq: mx.array,
    xk: mx.array,
    freqs_cis: mx.array,
    repeat_freqs_k: bool = False
) -> Tuple[mx.array, mx.array]:
    xq_ = view_as_complex(xq)
    xk_ = (view_as_complex(xk)
           if xk.shape[-2] != 0
           else None
    )
    
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        return xq_out.astype(xq.dtype), xk
    
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        # TODO: Check this implementation
        # MLX's mx.repeat works differently from PyTorch's repeat.
        # We need to use mx.expand_dims + mx.broadcast_to or mx.tile.
        # Given we want to repeat along the penultimate axis (usually sequence/RoPE axis)
        # Construct the repeating shape explicitly:
        reps = [1] * (freqs_cis.ndim - 2) + [r, 1]
        freqs_cis = mx.tile(freqs_cis, reps)
    xk_out = real(xk_ * freqs_cis).flatten(3)
    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)
    
def window_partition(x: mx.array, window_size: int) -> Tuple[mx.array, Tuple[int, int]]:
    B, H, W, C = x.shape
    
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = mx.pad(x, (
            (0, 0),
            (0, pad_h),
            (0, pad_w),
            (0, 0),
        ))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.reshape(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)

def window_unpartition(
    windows: mx.array, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> mx.array:
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.transpose(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    
    return x

def get_abs_pos(
    abs_pos: mx.array,
    has_cls_token: bool,
    hw: Tuple[int, int],
    retain_cls_token: bool = False,
    tiling: bool = False,
) -> mx.array:
    if retain_cls_token:
        assert has_cls_token
    
    h, w = hw
    if has_cls_token:
        cls_pos = abs_pos[:, :1]
        abs_pos = abs_pos[:, 1:]
    
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = abs_pos.reshape(1, size, size, -1).transpose(0, 3, 1, 2)
        if tiling:
            current_h, current_w = new_abs_pos.shape[2], new_abs_pos.shape[3]
            rep_h = (h //  current_h) + 1
            rep_w = (w // current_w) + 1
            
            new_abs_pos = mx.tile(new_abs_pos, (1, 1, rep_h, rep_w))
            new_abs_pos = new_abs_pos[:, :, :h, :w]
        else:
            current_h, current_w = new_abs_pos.shape[1], new_abs_pos.shape[2] 

            scale_h = h / current_h
            scale_w = w / current_w
            # TODO: Using linear (bilinear) interpolation, CUBIC interpolation in torch SAM3
            upsample_fn = nn.Upsample(
                scale_factor=(scale_h, scale_w), 
                mode='cubic', 
                align_corners=False
            )
            new_abs_pos = upsample_fn(new_abs_pos.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
        
        if not retain_cls_token:
            return new_abs_pos.transpose(0, 2, 3, 1)
        else:
            assert has_cls_token
            return mx.concat(
                [cls_pos, new_abs_pos.transpose(0, 2, 3, 1).reshape(1, h * w, -1)],
                dim=1
            )
    else:
        if not retain_cls_token:
            return abs_pos.reshape(1, h, w, -1)
        else:
            assert has_cls_token
            return mx.concat([cls_pos, abs_pos], dim=1)
    
    
    
class PatchEmbed(nn.Module):
    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
        bias: bool = True,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    
    def __call__(self, x: mx.array) -> mx.array:
        # B C H W -> B H W C
        x = x.transpose(0, 2, 3, 1)
        x = self.proj(x)
        
        return x
        
class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
        cls_token: bool = True,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        rope_pt_size: Optional[Tuple[int, int]] = None,
        rope_interp: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.cls_token = cls_token

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # rel_pos embeddings and rope
        self.use_rel_pos = use_rel_pos
        self.input_size = input_size

        self.use_rope = use_rope
        self.rope_theata = rope_theta
        self.rope_pt_size = rope_pt_size
        self.rope_interp = rope_interp

        # init rel pos embeddings and rope
        self._setup_rel_pos(rel_pos_zero_init)
        self._setup_rope_freqs()
        
    def _setup_rel_pos(self, rel_pos_zero_init: bool = True) -> None:
        if not self.use_rel_pos:
            self.rel_pos_h = None
            self.rel_pos_w = None
            return
        
        assert self.input_size is not None
        assert self.cls_token is False, "not supported"
        # initialize relative positional embeddings
        self.rel_pos_h = mx.zeros((2 * self.input_size[0] - 1, self.head_dim))
        self.rel_pos_w = mx.zeros((2 * self.input_size[1] - 1, self.head_dim))

        if not rel_pos_zero_init:
            # TODO: In pytorch initialization is done with nn.init.trunc_normal_
            self.rel_pos_h = mx.random.truncated_normal(lower=-2, uppper=2, shape=self.rel_pos_h.shape) * 0.02
            self.rel_pos_w = mx.random.truncated_normal(lower=-2, uppper=2, shape=self.rel_pos_w.shape) * 0.02
        
        # Precompute the relative coords
        H, W = self.input_size
        q_coords = mx.arange(H)[:,None]
        k_coords = mx.arange(W)[None, :]
        relative_coords = (q_coords - k_coords) + (H - 1)
        self.relative_coords = relative_coords.astype(mx.int64)
            
    def _setup_rope_freqs(self) -> None:
        if not self.use_rope:
            self.freqs_cis = None
            return
        
        assert self.input_size is not None

        if self.rope_pt_size is None:
            self.rope_pt_size = self.input_size
        
        self.compute_cis = partial(
            compute_axial_cis,
            dim=self.head_dim,
            theta=self.rope_theata,
        )

        scale_pos = 1.0
        if self.rope_interp:
            scale_pos = self.rope_pt_size[0] / self.input_size[0]
        # get scaled freqs_cis
        freqs_cis = self.compute_cis(
            end_x=self.input_size[0],
            end_y=self.input_size[1],
            scale_pos=scale_pos,
        )
        
        if self.cls_token:
            t = mx.zeros(
                self.head_dim // 2,
                dtype=mx.float32,
            )
            cls_freqs_cis = polar(mx.ones_like(t), t)[None, :]
            freqs_cis = mx.concat([cls_freqs_cis, freqs_cis], axis=0)
        
        self.freqs_cis = freqs_cis
        
    def _apply_rope(self, q, k) -> Tuple[mx.array, mx.array]:
        if not self.use_rope:
            return q, k

        assert self.freqs_cis is not None
        return apply_rotary_enc(q, k, freqs_cis=self.freqs_cis)
    
    def __call__(self, x: mx.array) -> mx.array:
        s = 1 if self.cls_token else 0
        if x.ndim == 4:
            B, H, W, _ = x.shape
            assert s == 0
            L = H * W
            ndim = 4
        else:
            assert x.ndim == 3
            B, L, _ = x.shape
            ndim = 3
            H = W = math.sqrt(L - s)
        
        # qkv with shape (3, B, nHead, L, C)
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, -1)
        # q, k, v with shape (B, nHead, L, C)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # handle rope and pos embeddings
        q, k = self._apply_rope(q, k)
        if self.use_rel_pos:
            # TODO: Skipping relative positional embeddings for now
            pass
            
        scale = q.shape[-1] ** -0.5
        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        
        if ndim == 4:
            new_shape = (B, self.num_heads, H, W, -1)
            x = (
                x.reshape(new_shape)
                .transpose(0, 2, 3, 1, 4)
                .reshape(B, H, W, -1)
            )
        else:
            new_shape = (B, L, self.num_heads, -1)
            x = x.reshape(new_shape).transpose(0, 2, 1, 3).reshape(B, L, -1)
        
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        use_rope: bool = False,
        rope_pt_size: Optional[Tuple[int, int]] = None,
        rope_tiled: bool = False,
        rope_interp: bool = False,
        use_ve_rope: bool = False,
        cls_token: bool = False,
        dropout: float = 0.0,
        init_values: Optional[float] = None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            use_rope=use_rope,
            rope_pt_size=rope_pt_size,
            rope_interp=rope_interp,
            cls_token=cls_token,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=(dropout, 0.0),
        )
        
        # Confused if it's the right way to do since, I'm not doing an inplace operation in LayerScale
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)
        self.window_size = window_size

    def __call__(self, x: mx.array) -> mx.array:
        shortcut = x
        x = self.norm1(x)
        
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
       
        x = self.ls1(self.attn(x))
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        
        x = shortcut + self.dropout(self.drop_path(x))
        x = x + self.dropout(self.drop_path(self.ls2(self.mlp(self.norm2(x)))))
        
        return x

class ViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        norm_layer: Union[Callable[..., nn.Module], str] = "LayerNorm",
        act_layer: Callable[..., nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        tile_abs_pos: bool = True,
        rel_pos_blocks: Union[Tuple[int, ...], bool] = (2, 5, 8, 11),
        rel_pos_zero_init: bool = True,
        window_size: int = 14,
        global_att_blocks: Tuple[int, ...] = (2, 5, 8, 11),
        use_rope: bool = False,
        rope_pt_size: Optional[int] = None,
        use_interp_rope: bool = False,
        pretrain_img_size: int = 224,
        pretrain_use_cls_token: bool = True,
        retain_cls_token: bool = True,
        dropout: float = 0.0,
        return_interm_layers: bool = False,
        init_values: Optional[float] = None,
        ln_pre: bool = False,
        ln_post: bool = False,
        bias_patch_embed: bool = True,
        compile_mode: Optional[str] = None,
        use_act_checkpoint: bool = True,
    ):
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token

        window_block_indexes = [i for i in range(depth) if i not in global_att_blocks]
        self.full_attn_ids = list(global_att_blocks)
        self.rel_pos_blocks = [False] * depth
        if isinstance(rel_pos_blocks, bool) and rel_pos_blocks:
            self.rel_pos_blocks = [True] * depth
        else:
            for i in rel_pos_blocks:
                self.rel_pos_blocks[i] = True
        
        self.retain_cls_token = retain_cls_token
        if self.retain_cls_token:
            assert pretrain_use_cls_token
            assert (
                len(window_block_indexes) == 0
            ), "windowing not supported with cls token"
            assert sum(self.rel_pos_blocks) == 0, "rel pos not supported with cls token"

            scale = embed_dim ** -0.5
            self.class_embedding = nn.Parameter(scale * mx.random.normal((1, 1, embed_dim)))
        
        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-5)
        
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=bias_patch_embed,
        )

        # handle absolute position embedding
        self.tile_abs_pos = tile_abs_pos
        self.use_abs_pos = use_abs_pos
        if self.tile_abs_pos:
            assert self.use_abs_pos
        
        if self.use_abs_pos:
            num_patches = (pretrain_img_size // patch_size) * (
                pretrain_img_size // patch_size
            )
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            # TODO: Check Clearly about nn.Parameter and how mlx handles trainable and non-trainable parameters
            self.pos_embed = mx.zeros((1, num_positions, embed_dim))
        else:
            self.pos_embed = None
            
        # stochastic depth decay rule
        dpr = [x.item() for x in mx.linspace(0, drop_path_rate, depth)]

        self.blocks = []
        cur_stage = 1
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=self.rel_pos_blocks[i],
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                use_rope=use_rope,
                rope_pt_size=(
                    (window_size, window_size)
                    if rope_pt_size is None
                    else (rope_pt_size, rope_pt_size)
                ),
                rope_interp=use_interp_rope,
                cls_token=self.retain_cls_token,
                dropout=dropout,
                init_values=init_values,
            )

            if i not in window_block_indexes:
                cur_stage += 1
            
            self.use_act_checkpoint = use_act_checkpoint
            self.blocks.append(block)

        self.window_block_indexes = window_block_indexes
        self.return_interm_layers = return_interm_layers
        self.channel_list = (
            [embed_dim] * len(self.full_attn_ids)
            if return_interm_layers
            else [embed_dim]
        )

        if self.pos_embed is not None:
            self.pos_embed = mx.random.truncated_normal(lower=-2, upper=2, shape=self.pos_embed.shape) * 0.02
        
        self.ln_pre = norm_layer(embed_dim) if ln_pre else nn.Identity()
        self.ln_post = norm_layer(embed_dim) if ln_post else nn.Identity()
        
        self._init_weights(self)

        # TODO: handing compiled mode
        
    
    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            m.weight = mx.random.truncated_normal(lower=-2, upper=2, shape=m.weight.shape) * 0.02
            if m.bias is not None:
                m.bias = mx.zeros_like(m.bias)
        elif isinstance(m, nn.LayerNorm):
            m.weight = mx.ones_like(m.weight)
            m.bias = mx.zeros_like(m.bias)

    def __call__(self, x: mx.array) -> List[mx.array]:
        x = self.patch_embed(x)
        h, w = x.shape[1], x.shape[2]

        s = 0
        if self.retain_cls_token:
            x = mx.concat([self.class_embedding, x.flatten(1, 2)], dim=1)
            s = 1
        
        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed,
                self.pretrain_use_cls_token,
                (h, w),
                self.retain_cls_token,
                tiling=self.tile_abs_pos,
            )
        
        x = self.ln_pre(x)
        
        outputs = []
        for i, blk in enumerate(self.blocks):
            if self.use_act_checkpoint and self.training:
                # TODO: impelement checkpointing
                pass
            else:
                x = blk(x)
                
            if (i == self.full_attn_ids[-1]) or (
                self.return_interm_layers and i in self.full_attn_ids
            ):
                if i == self.full_attn_ids[-1]:
                    x = self.ln_post(x)
                
                feats = x[:, s:]
                if feats.ndim == 4:
                    feats = feats.transpose(0, 3, 1, 2)
                else:
                    assert feats.ndim == 3
                    h = w = math.sqrt(feats.shape[1])
                    feats = feats.reshape(
                        feats.shape[0], h, w, feats.shape[-1]
                    ).transpose(0, 3, 1, 2)
                
                outputs.append(feats)
            
        return outputs
                        

        

    def get_layer_id(self, layer_name: str) -> int:
        # TODO: Implement this method
        pass

    def get_num_layers(self) -> int:
        return len(self.blocks)
        