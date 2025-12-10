import mlx.nn as nn
import mlx.core as mx
from mlx.utils import tree_map_with_path
from sam3.model.maskformer_segmentation import PixelDecoder
from sam3.model_builder import build_sam3_image_model, _create_segmentation_head, _create_transformer_decoder, _create_dot_product_scoring, _create_vision_backbone

def main():
    # checkpoint_path = "/Users/deekshith/Documents/Projects/vision-models/mlx_sam3/sam3-mod-weights/model.safetensors"
    # build_sam3_image_model(
    #     checkpoint_path=checkpoint_path,
    # )

    
    seg_head = _create_segmentation_head()
    inputs = {
        "backbone_feats": [
            mx.random.normal((1, 256, 288, 288)),
            mx.random.normal((1, 256, 144, 144)),
            mx.random.normal((1, 256, 72, 72)),
            ],
        "obj_queries": mx.random.normal((6, 1, 200, 256)),
        "image_ids": mx.array([0], dtype=mx.int64),
        "encoder_hidden_states": mx.random.normal((5184, 1, 256)),
        "prompt": mx.random.normal((33, 1, 256)),
        "prompt_mask": mx.random.normal((1,33)) > 0.5
    }

    out = seg_head(**inputs)
    
    # prompt_mask = mx.random.normal((1, 33)) > 0.5
    # model = _create_dot_product_scoring()
    # input = {
    #     "hs": mx.random.normal((6 ,1, 200, 256)),
    #     "prompt": mx.random.normal((33, 1, 256)),
    #     "prompt_mask": prompt_mask,
    # }

    # output = model(**input)

if __name__ == "__main__":
    main()
