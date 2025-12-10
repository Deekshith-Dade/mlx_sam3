from copy import deepcopy
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class Scale4FN(nn.Module):
    def __init__(self, in_channels: int, d_model: int, use_bias: bool = True):
        super().__init__()
        self.dconv_2x2_0 = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )
        self.gelu = nn.GELU()
        self.dconv_2x2_1 = nn.ConvTranspose2d(
            in_channels // 2,
            in_channels // 4,
            kernel_size=2,
            stride=2
        )
        self.conv_1x1 = nn.Conv2d(
            in_channels=in_channels // 4,
            out_channels=d_model,
            kernel_size=1,
            bias=use_bias
        )
        self.conv_3x3 = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            bias=use_bias
        )

    def __call__(self, x):
        x = x.transpose(0, 2, 3, 1)
        x = self.dconv_2x2_0(x)
        x = self.gelu(x)
        x = self.dconv_2x2_1(x)
        x = self.conv_1x1(x)
        x = self.conv_3x3(x)
        x = x.transpose(0, 3, 1, 2)
        return x

class Scal2FN(nn.Module):
    def __init__(self, in_channels: int, d_model: int, use_bias: bool = True):
        super().__init__()
        self.dconv_2x2 = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )
        self.gelu = nn.GELU()
        self.conv_1x1 = nn.Conv2d(
            in_channels=in_channels // 2,
            out_channels=d_model,
            kernel_size=1,
            bias=use_bias
        )
        self.conv_3x3 = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            bias=use_bias
        )

    def __call__(self, x):
        x = x.transpose(0, 2, 3, 1)
        x = self.dconv_2x2(x)
        x = self.gelu(x)
        x = self.conv_1x1(x)
        x = self.conv_3x3(x)
        x = x.transpose(0, 3, 1, 2)
        return x

class Scale1FN(nn.Module):
    def __init__(self, in_channels: int, d_model: int, use_bias: bool = True):
        super().__init__()
        self.conv_1x1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=1,
            bias=use_bias
        )
        self.conv_3x3 = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            bias=use_bias
        )

    def __call__(self, x):
        x = x.transpose(0, 2, 3, 1)
        x = self.conv_1x1(x)
        x = self.conv_3x3(x)
        x = x.transpose(0, 3, 1, 2)
        return x

class Scale0_5FN(nn.Module):
    def __init__(self, in_channels: int, d_model: int, use_bias: bool = True):
        super().__init__()
        self.maxpool_2x2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.conv_1x1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=1,
            bias=use_bias
        )
        self.conv_3x3 = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            bias=use_bias
        )

    def __call__(self, x):
        x = x.transpose(0, 2, 3, 1)
        x = self.maxpool_2x2(x)
        x = self.conv_1x1(x)
        x = self.conv_3x3(x)
        x = x.transpose(0, 3, 1, 2)
        return x

class Sam3DualViTDetNeck(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        position_encoding: nn.Module,
        d_model: int,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        add_sam2_neck: bool = False
        
    ):
        super().__init__()
        self.trunk = trunk
        self.position_encoding = position_encoding
        self.convs = []

        self.scale_factors = scale_factors
        use_bias = True
        dim: int = self.trunk.channel_list[-1]

        self.convs = self._build_convs(dim, d_model, scale_factors, use_bias)
        
        self.sam2_convs = None
        if add_sam2_neck:
            self.sam2_convs = self._build_convs(dim, d_model, scale_factors, use_bias)

    def _build_convs(self, dim, d_model, scale_factors, use_bias):
        convs = []
        for _, scale in enumerate(scale_factors):
            if scale == 4.0:
                convs.append(
                    Scale4FN(
                        in_channels=dim,
                        d_model=d_model,
                        use_bias=use_bias
                ))
            elif scale == 2.0:
                convs.append(
                    Scal2FN(
                        in_channels=dim,
                        d_model=d_model,
                        use_bias=use_bias
                ))
            elif scale == 1.0:
                convs.append(
                    Scale1FN(
                        in_channels=dim,
                        d_model=d_model,
                        use_bias=use_bias
                ))
            elif scale == 0.5:
                convs.append(
                    Scale0_5FN(
                        in_channels=dim,
                        d_model=d_model,
                        use_bias=use_bias
                ))
            else:
                raise NotImplementedError(f"Scale factor {scale} not supported yet.")
        return convs
        
    def __call__(
        self, x_list: List[mx.array]
    ) -> Tuple[
        List[mx.array],
        List[mx.array],
        Optional[List[mx.array]],
        Optional[List[mx.array]],
    ]:
        xs = self.trunk(x_list)
        sam3_out, sam3_pos = [], []
        sam2_out, sam2_pos = None, None       
        if self.sam2_convs is not None:
            sam2_out, sam2_pos = [], []
        x = xs[-1]
        for i in range(len(self.convs)):
            sam3_x_out = self.convs[i](x)
            sam3_pos_out = self.position_encoding(sam3_x_out).astype(sam3_x_out.dtype)
            sam3_out.append(sam3_x_out)
            sam3_pos.append(sam3_pos_out)

            if self.sam2_convs is not None:
                sam2_x_out = self.sam2_convs[i](x)
                sam2_pos_out = self.position_encoding(sam2_x_out).astype(sam2_x_out.dtype)
                sam2_out.append(sam2_x_out)
                sam2_pos.append(sam2_pos_out)
        return sam3_out, sam3_pos, sam2_out, sam2_pos