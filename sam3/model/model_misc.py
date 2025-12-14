import math
from functools import partial
from typing import Optional, Tuple, Type, Union

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path

def inverse_sigmoid(x, eps=1e-3):
    x = mx.clip(x, 0, 1)
    x1 = mx.clip(x, eps, None)
    x2 = mx.clip((1-x), eps, None)
    return mx.log(x1 / x2)


class MultiheadAttentionWrapper(nn.MultiHeadAttention):
    def __init__(self, *args, **kwargs):
        kwargs["bias"] = True
        super().__init__(*args, **kwargs)
    def __call__(self, *args, **kwargs):

        if kwargs.get('attn_mask', None) is None:
            kwargs['attn_mask'] = None
        if kwargs.get('key_padding_mask', None) is None:
            kwargs['key_padding_mask'] = None

        key_padding_mask = kwargs['key_padding_mask']
        attn_mask = kwargs['attn_mask']

        # TODO: check this implementation from codex5.1
        # Convert key padding mask (bs, src_len) to additive mask
        padding_mask = None
        if key_padding_mask is not None:
            padding_mask = mx.where(key_padding_mask, -float('inf'), 0.0)
            padding_mask = padding_mask[:, None, None, :]

        # Normalize attn_mask to additive form compatible with mlx MultiHeadAttention
        normalized_attn_mask = None
        if attn_mask is not None:
            mask = attn_mask
            # Bring bool masks to additive 0 / -inf form
            if mask.dtype == mx.bool_:
                mask = mx.where(mask, -float('inf'), 0.0)

            if mask.ndim == 2:
                # (tgt_len, src_len) -> (1, 1, tgt_len, src_len)
                mask = mask[None, None, :, :]
            elif mask.ndim == 3:
                # Accept (bs, tgt_len, src_len) or (bs * num_heads, tgt_len, src_len)
                q = args[0] if len(args) > 0 else kwargs.get("query") or kwargs.get("queries")
                batch_size = q.shape[0] if q is not None else None
                tgt_len, src_len = mask.shape[-2], mask.shape[-1]
                if batch_size is not None:
                    if mask.shape[0] == batch_size * self.num_heads:
                        mask = mask.reshape(batch_size, self.num_heads, tgt_len, src_len)
                    elif mask.shape[0] == batch_size:
                        mask = mask[:, None, :, :]
                # else: assume already (batch, heads, tgt, src) or compatible
            normalized_attn_mask = mask

        # Combine padding and attention masks
        if padding_mask is not None and normalized_attn_mask is not None:
            final_mask = padding_mask + normalized_attn_mask
        elif padding_mask is not None:
            final_mask = padding_mask
        else:
            final_mask = normalized_attn_mask

        kwargs.pop('attn_mask', None)
        kwargs.pop('key_padding_mask', None)
        kwargs['mask'] = final_mask

        return super().__call__(*args, **kwargs)


class DotProductScoring(nn.Module):
    def __init__(
        self,
        d_model,
        d_proj,
        prompt_mlp=None,
        clamp_logits=True,
        clamp_max_val=12.0,
    ):
        super().__init__()
        self.d_proj = d_proj
        assert isinstance(prompt_mlp, nn.Module) or prompt_mlp is None
        self.prompt_mlp = prompt_mlp  # an optional MLP projection for prompt
        self.prompt_proj = nn.Linear(d_model, d_proj)
        self.hs_proj = nn.Linear(d_model, d_proj)
        self.scale = float(1.0 / np.sqrt(d_proj))
        self.clamp_logits = clamp_logits
        if self.clamp_logits:
            self.clamp_max_val = clamp_max_val
    
    def mean_pool_text(self, prompt, prompt_mask):
        # is_valid has shape (seq, bs, 1), where 1 is valid and 0 is padding
        is_valid = (~prompt_mask).astype(mx.float32).transpose(1, 0)[..., None]
        # num_valid has shape (bs, 1)
        num_valid = mx.clip(mx.sum(is_valid, axis=0), 1.0, None)
        # mean pool over all the valid tokens -- pooled_prompt has shape (bs, proj_dim)
        pooled_prompt = mx.sum(prompt * is_valid, axis=0) / num_valid
        return pooled_prompt

    def __call__(self, hs, prompt, prompt_mask):
        # hs has shape (num_layer, bs, num_query, d_model)
        # prompt has shape (seq, bs, d_model)
        # prompt_mask has shape (bs, seq), where 1 is valid and 0 is padding
        assert hs.ndim == 4 and prompt.ndim == 3 and prompt_mask.ndim == 2

        # apply MLP on prompt if specified
        if self.prompt_mlp is not None:
            prompt = self.prompt_mlp(prompt)

        # first, get the mean-pooled version of the prompt
        pooled_prompt = self.mean_pool_text(prompt, prompt_mask)

        # then, project pooled_prompt and hs to d_proj dimensions
        proj_pooled_prompt = self.prompt_proj(pooled_prompt)  # (bs, d_proj)
        proj_hs = self.hs_proj(hs)  # (num_layer, bs, num_query, d_proj)

        # finally, get dot-product scores of shape (num_layer, bs, num_query, 1)
        scores = mx.matmul(proj_hs, proj_pooled_prompt[..., None])
        scores *= self.scale

        # clamp scores to a max value to avoid numerical issues in loss or matcher
        if self.clamp_logits:
            scores = mx.clip(scores, -self.clamp_max_val, self.clamp_max_val)

        return scores

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    # TODO: attribute timm for implementation reference
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = mx.random.bernoulli(p=keep_prob, shape=shape)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor = random_tensor / keep_prob
    return x * random_tensor


class DropPath(nn.Module):
    # TODO: attribute timm for implementation reference
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def __call__(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class TransformerWrapper(nn.Module):
    def __init__(
        self,
		encoder,
		decoder,
		d_model: int,
		two_stage_type="none",
		pos_enc_at_input_dec=True,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.num_queries = decoder.num_queries if decoder is not None else None
        self.pos_enc_at_input_dec = pos_enc_at_input_dec

        # for two stage
        assert two_stage_type in ["none"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        self.two_stage_type = two_stage_type

        self._reset_parameters()
        self.d_model = d_model
    
    def _reset_parameters(self):
        def _init_fn(path, params):
            if params.ndim > 1:
                if (
                "box_embed" not in path
                and "query_embed" not in path
                and "reference_points" not in path
            ):
                    return nn.init.glorot_uniform()(params, 1.0)
            return params
 
        self.update(tree_map_with_path(_init_fn, self.parameters()))
            
class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        residual: bool = False,
        out_norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = [
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        ]
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # whether to add the output as a residual connection to the input
        if residual and input_dim != output_dim:
            raise ValueError("residual is only supported if input_dim == output_dim")
        self.residual = residual
        # whether to apply a normalization layer to the output
        assert isinstance(out_norm, nn.Module) or out_norm is None
        self.out_norm = out_norm or nn.Identity()
        self.act = nn.ReLU()

    def __call__(self, x):
        orig_x = x
        for i, layer in enumerate(self.layers):
            x = self.drop(self.act(layer(x))) if i < self.num_layers - 1 else layer(x)
        if self.residual:
            x = x + orig_x
        x = self.out_norm(x)
        return x
    
class Mlp(nn.Module):
    # TODO: attribute timm for implementation reference
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Optional[Type[nn.Module]] = None,
        bias: Union[bool, Tuple[bool, bool]] = True,
        drop: Union[float, Tuple[float, float]] = 0.,
        use_conv: bool = False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = bias if isinstance(bias, tuple) else (bias, bias)
        drop_probs = drop if isinstance(drop, tuple) else (drop, drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
    
    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
        
        
        
class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, mx.array] = 1e-5,
        inplace: bool = False
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = init_values * mx.ones(dim)
    
    def __call__(self, x: mx.array) -> mx.array:
        # Note: MLX arrays are immutable, so "inplace" operations still create new arrays.
        # The inplace flag is kept for API compatibility with PyTorch but doesn't change behavior.
        # Both paths return a new array.
        return x * self.gamma



def get_clones(module, N):
    return [module() for i in range(N)]

def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.relu
    if activation == "gelu":
        return nn.gelu
    if activation == "glu":
        return nn.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

def get_valid_ratio(mask):
    _, H, W = mask.shape
    valid_H = mx.sum(~mask[:, :, 0], 1)
    valid_W = mx.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W
    valid_ratio = mx.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio

def gen_sineembed_for_position(pos_array, num_feats=256):
    assert num_feats % 2 == 0
    num_feats = num_feats // 2
    
    scale = 2 * math.pi
    dim_t = mx.arange(num_feats, dtype=mx.float32)
    # TODO: rounding mode?
    dim_t = 10000 ** (2 * mx.floor(mx.divide(dim_t, 2)) / num_feats)
    x_embed = pos_array[:, :, 0] * scale
    y_embed = pos_array[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = mx.stack(
        (mx.sin(pos_x[:, :, 0::2]), mx.cos(pos_x[:, :, 1::2])), axis=3
    ).flatten(2)
    pos_y = mx.stack(
        (mx.sin(pos_y[:, :, 0::2]), mx.cos(pos_y[:, :, 1::2])), axis=3
    ).flatten(2)
    if pos_array.shape[-1] == 2:
        pos = mx.concat([pos_y, pos_x], axis=2)
    elif pos_array.shape[-1] == 4:
        w_embed = pos_array[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = mx.stack(
            (mx.sin(pos_w[:, :, 0::2]), mx.cos(pos_w[:, :, 1::2])), axis=3
        ).flatten(2)

        h_embed = pos_array[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = mx.stack(
            (mx.sin(pos_h[:, :, 0::2]), mx.cos(pos_h[:, :, 1::2])), axis=3
        ).flatten(2)

        pos = mx.concat((pos_y, pos_x, pos_w, pos_h), axis=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(mx.shape[-1]))
    return pos    
