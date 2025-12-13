from dataclasses import dataclass 
import mlx.core as mx
import mlx.nn as nn

from typing import Any, get_args, get_origin, List, Mapping, Optional, Sequence, Union

MyTensor = Union[mx.array, List[Any]]

def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    if input.size == 0:
        out_shape = list(input.shape)
        if size is not None:
            # size is usually (H, W)
            out_shape[2] = size[0] 
            out_shape[3] = size[1]
        elif scale_factor is not None:
            out_shape[2] = int(out_shape[2] * scale_factor)
            out_shape[3] = int(out_shape[3] * scale_factor)
        return mx.zeros(out_shape, dtype=input.dtype)
    
    x = input.transpose(0, 2, 3, 1)
    if mode == 'bilinear' or mode == 'bicubic':
        mode = 'linear'
    
    current_h, current_w = x.shape[1], x.shape[2]

    if size is not None:
        if isinstance(size, int):
            size = (size, size)
        
        scale_h = size[0] / current_h
        scale_w = size[1] / current_w
        final_scale = (scale_h, scale_w)
    
    elif scale_factor is not None:
        if isinstance(scale_factor, (float, int)):
            final_scale = (float(scale_factor), float(scale_factor))
        else:
            final_scale = scale_factor
            
    else:
        raise ValueError("Either size or scale_factor must be defined")

    upsample_layer = nn.Upsample(
        scale_factor=final_scale, 
        mode=mode,
        align_corners=align_corners
    )
    
    x = upsample_layer(x)

    return x.transpose(0, 3, 1, 2)

@dataclass
class FindStage:
    img_ids: MyTensor
    img_ids__type = mx.int64
    text_ids: MyTensor
    text_ids__type = mx.int64

    input_boxes: MyTensor
    input_boxes__type = mx.float32
    input_boxes_mask: MyTensor
    input_boxes_mask__type = mx.bool_
    input_boxes_label: MyTensor
    input_boxes_label__type = mx.int64

    input_points: MyTensor
    input_points__type = mx.float32
    input_points_mask: MyTensor
    input_points_mask__type = mx.bool_

    # We track the object ids referred to by this query.

    # This is beneficial for tracking in videos without the need for pointers.
    object_ids: Optional[List[List]] = None  # List of objects per query