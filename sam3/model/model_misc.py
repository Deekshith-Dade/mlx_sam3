import math
import copy
from functools import partial
from typing import Optional, Tuple, Type, Union
import mlx.core as mx
import mlx.nn as nn

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

        padding_mask = None
        if key_padding_mask is not None:
            padding_mask = mx.where(key_padding_mask, -float('inf'), 0.0)
            padding_mask = padding_mask[:, None, None, :]

        final_mask = padding_mask
        if attn_mask is not None:
            # TODO: check this
            if attn_mask.ndim == 2:
                attn_mask = attn_mask[None, None, :, :]
                final_mask = final_mask + attn_mask
        
        
        del kwargs['attn_mask']
        del kwargs['key_padding_mask']
        kwargs['mask'] = final_mask

        return super().__call__(*args, **kwargs)


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
    dim_t = 10000 * (2 * (mx.divide(dim_t, 2)) / num_feats)
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
