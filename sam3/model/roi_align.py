from typing import Union
import mlx.core as mx

# NB: all inputs are tensors
def _bilinear_interpolate(
    input,  # [N, C, H, W]
    roi_batch_ind,  # [K]
    y,  # [K, PH, IY]
    x,  # [K, PW, IX]
    ymask,  # [K, IY]
    xmask,  # [K, IX]
):
    _, channels, height, width = input.shape
    # deal with inverse element out of feature map boundary
    y = mx.clip(y, a_min=0, a_max=None)
    x = mx.clip(x, a_min=0, a_max=None)

    y_low = y.astype(mx.int32)
    x_low = x.astype(mx.int32)
    y_high = mx.where(y_low >= height - 1, height - 1, y_low + 1)
    y_low = mx.where(y_low >= height - 1, height - 1, y_low)
    y = mx.where(y_low >= height - 1, y.astype(input.dtype), y)

    x_high = mx.where(x_low >= width - 1, width - 1, x_low + 1)
    x_low = mx.where(x_low >= width - 1, width - 1, x_low)
    x = mx.where(x_low >= width - 1, x.astype(input.dtype), x)

    ly = y - y_low
    lx = x - x_low
    hy = 1.0 - ly
    hx = 1.0 - lx
    
    ly = y - y_low

    # do bilinear interpolation, but respect the masking!
    # TODO: It's possible the masking here is unnecessary if y and
    # x were clamped appropriately; hard to tell
    def masked_index(
        y,  # [K, PH, IY]
        x,  # [K, PW, IX]
    ):
        if ymask is not None:
            assert xmask is not None
            y = mx.where(ymask[:, None, :], y, 0)
            x = mx.where(xmask[:, None, :], x, 0)
        return input[
            roi_batch_ind[:, None, None, None, None, None],
            mx.arange(channels)[None, :, None, None, None, None],
            y[:, None, :, None, :, None],  # prev [K, PH, IY]
            x[:, None, None, :, None, :],  # prev [K, PW, IX]
        ]  # [K, C, PH, PW, IY, IX]
    
    v1 = masked_index(y_low, x_low)
    v2 = masked_index(y_low, x_high)
    v3 = masked_index(y_high, x_low)
    v4 = masked_index(y_high, x_high)
    
    # all ws preemptively [K, C, PH, PW, IY, IX]
    def outer_prod(y, x):
        return y[:, None, :, None, :, None] * x[:, None, None, :, None, :]

    w1 = outer_prod(hy, hx)
    w2 = outer_prod(hy, lx)
    w3 = outer_prod(ly, hx)
    w4 = outer_prod(ly, lx)

    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    return val

def convert_boxes_to_roi_format(boxes: list[mx.array]) -> mx.array:
    concat_boxes = mx.concat([b for b in boxes], axis=0)
    temp = []
    for i, b in enumerate(boxes):
        temp.append(mx.full(b[:, :1].shape, i))
    ids = mx.concat(temp, axis=0)
    rois = mx.concat([ids, concat_boxes], axis=1)
    return rois

def check_roi_boxes_shape(boxes: Union[mx.array, Union[list[mx.array], tuple[mx.array]]]):
    if isinstance(boxes, (list, tuple)):
        for _array in boxes:
            assert _array.shape[1] == 4, "The shape of the tensor in the boxes list is not correct as List[Tensor[L, 4]]"
            
    elif isinstance(boxes, mx.array):
        assert boxes.shape[1] == 5, "The boxes tensor shape is not correct as Tensor[K, 5]"
    return

def _roi_align(
   input,
   rois,
   spatial_scale,
   pooled_height,
   pooled_width,
   sampling_ratio,
   aligned 
):
    orig_dtype = input.dtype
    
    _, _, height, width = input.shape 

    ph = mx.arange(pooled_height)
    pw = mx.arange(pooled_width)

    # inputs: [N, C, H, W]
    # rois: [K, 5]

    roi_batch_ind = rois[:, 0].astype(mx.int32)
    offset = 0.5 if aligned else 0.0
    roi_start_w = rois[:, 1] * spatial_scale - offset  # [K]
    roi_start_h = rois[:, 2] * spatial_scale - offset  # [K]
    roi_end_w = rois[:, 3] * spatial_scale - offset  # [K]
    roi_end_h = rois[:, 4] * spatial_scale - offset  # [K]
    
    roi_width = roi_end_w - roi_start_w # [K]
    roi_height = roi_end_h - roi_start_h # [K]
    if not aligned:
        roi_width = mx.clip(roi_width, a_min=1.0, a_max=None)
        roi_height = mx.clip(roi_height, a_min=1.0, a_max=None)
    
    bin_size_h = roi_height / pooled_height  # [K]
    bin_size_w = roi_width / pooled_width  # [K]
    
    exact_sampling = sampling_ratio > 0
    
    roi_bin_grid_h = sampling_ratio if exact_sampling else mx.ceil(roi_height / pooled_height)
    roi_bin_grid_w = sampling_ratio if exact_sampling else mx.ceil(roi_width / pooled_width)

    if exact_sampling:
        count = max(roi_bin_grid_h * roi_bin_grid_w, 1)
        iy = mx.arange(roi_bin_grid_h)
        ix = mx.arange(roi_bin_grid_w)
        ymask = None
        xmask = None
    else:
        count = mx.clip(roi_bin_grid_h * roi_bin_grid_w, a_min=1, a_max=None)
        iy = mx.arange(height)
        ix = mx.arange(width)
        ymask = iy[None, :] < roi_bin_grid_h[:, None]
        xmask = ix[None, :] < roi_bin_grid_w[:, None]
    
    def from_K(t):
        return t[:, None, None]
    
    y = (
        from_K(roi_start_h)
        + ph[None, :, None] * from_K(bin_size_h)
        + (iy[None, None, :] + 0.5).astype(input.dtype) * from_K(bin_size_h / roi_bin_grid_h)
    )  # [K, PH, IY]
    x = (
        from_K(roi_start_w)
        + pw[None, :, None] * from_K(bin_size_w)
        + (ix[None, None, :] + 0.5).astype(input.dtype) * from_K(bin_size_w / roi_bin_grid_w)
    )  # [K, PW, IX]
    val = _bilinear_interpolate(input, roi_batch_ind, y, x, ymask, xmask)  # [K, C, PH, PW, IY, IX]
    
    if not exact_sampling:
        val = mx.where(ymask[:, None, None, None, :, None], val, 0)
        val = mx.where(xmask[:, None, None, None, None, :], val, 0)

    output = val.sum((-1, -2))  # remove IY, IX ~> [K, C, PH, PW]
    if isinstance(count, mx.array):
        output /= count[:, None, None, None]
    else:
        output /= count

    output = output.astype(orig_dtype)

    return output
    


def roi_align(
    input: mx.array,
    boxes: Union[mx.array, list[mx.array]],
    height: int,
    width: int,
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1, 
    aligned: bool = False,
):

    check_roi_boxes_shape(boxes)
    rois = convert_boxes_to_roi_format(boxes) if isinstance(boxes, (list, tuple)) else boxes
    return _roi_align(input, rois, spatial_scale, height, width, sampling_ratio, aligned)
