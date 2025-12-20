import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class PositionEmbeddingSine(nn.Module):
    def __init__(
        self,
        num_pos_feats,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
        precompute_resolution: Optional[int] = None,
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        
        self.cache = {}
        if precompute_resolution is not None:
            
            precompute_sizes = [
                (precompute_resolution // 4, precompute_resolution // 4),
                (precompute_resolution // 8, precompute_resolution // 8),
                (precompute_resolution // 16, precompute_resolution // 16),
                (precompute_resolution // 32, precompute_resolution // 32),
            ]
            for size in precompute_sizes:
                tensors = mx.zeros((1,1) + size)
                self(tensors)
    
    def _encode_xy(self, x, y):
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale
        
        dim_t = mx.arange(self.num_pos_feats, dtype=mx.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = mx.stack(
            (mx.sin(pos_x[:, 0::2]), mx.cos(pos_x[:, 1::2])),
            axis=2
        ).flatten(1)
        pos_y = mx.stack(
            (mx.sin(pos_y[:, 0::2]), mx.cos(pos_y[:, 1::2])),
            axis=2
        ).flatten(1)
        return pos_x, pos_y
    
    def encode_boxes(self, x, y, w, h):
        pos_x, pos_y = self._encode_xy(x, y)
        pos = mx.concat((pos_y, pos_x, h[:, None], w[:, None]), axis=1)
        return mx.stop_gradient(pos)
    
    encode = encode_boxes
    
    def encode_points(self, x, y, labels):
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
        pos = mx.concat((pos_y, pos_x, labels[:, :, None]), axis=2)
        return mx.stop_gradient(pos)
    
    def __call__(self, x: mx.array | tuple) -> mx.array:
        """
        Args:
            x: Either an mx.array (NCHW format) or a shape tuple (N, C, H, W)
        Returns:
            Position encoding in NCHW format
        """
        shape = x if isinstance(x, tuple) else x.shape
        batch, _, height, width = shape
        
        cache_key = (height, width)
        if cache_key in self.cache:
            return mx.repeat(self.cache[cache_key][None], repeats=batch, axis=0)
        
        y_embed = (
            mx.arange(1, height + 1, dtype=mx.float32)
            .reshape(1, -1, 1)
        )
        y_embed = mx.broadcast_to(y_embed, (batch, height, width))
        x_embed = (
            mx.arange(1, width + 1, dtype=mx.float32)
            .reshape(1, 1, -1)
        )
        x_embed = mx.broadcast_to(x_embed, (batch, height, width))

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        
        dim_t = mx.arange(self.num_pos_feats, dtype=mx.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = mx.stack(
            (mx.sin(pos_x[:, :, :, 0::2]), mx.cos(pos_x[:, :, :, 1::2])),
            axis=4
        ).flatten(3)
        pos_y = mx.stack(
            (mx.sin(pos_y[:, :, :, 0::2]), mx.cos(pos_y[:, :, :, 1::2])),
            axis=4
        ).flatten(3)
        pos = mx.concat((pos_y, pos_x), axis=3).transpose(0, 3, 1, 2)
        if cache_key is not None:
            self.cache[cache_key] = pos[0]
        return mx.stop_gradient(pos)

            