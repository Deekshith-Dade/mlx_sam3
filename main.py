import mlx.core as mx
from sam3.model_builder import build_sam3_image_model, _create_transformer_encoder, _create_transformer_decoder

def main():
    # checkpoint_path = "/Users/deekshith/Documents/Projects/vision-models/mlx_sam3/sam3-mod-weights/model.safetensors"
    # build_sam3_image_model(
    #     checkpoint_path=checkpoint_path,
    # )
    shape = (1, 33)
    prompt_mask = mx.random.uniform(shape=shape) < 0.5
    inputs = {
        "src": [mx.random.normal((5184, 1, 256))],
        "src_key_padding_mask": None,
        "src_pos": [mx.random.normal((5184, 1, 256))],
        "prompt": mx.random.normal((33, 1, 256)),
        "prompt_pos": mx.random.normal((33, 1, 256)),
        "prompt_key_padding_mask": prompt_mask,
        "feat_sizes": [(72, 72)]

    }
    encoder = _create_transformer_encoder()
    decoder = _create_transformer_decoder()
    out = encoder(**inputs)
    
    inputs = {
        "tgt": mx.random.normal((200, 1, 256)),
        "memory": mx.random.normal((5184, 1, 256)),
        "memory_key_padding_mask": None,
        "pos": mx.random.normal((5184, 1, 256)),
        "reference_boxes":None,
        "level_start_index":mx.random.randint(0, 1, (1,), dtype=mx.int32),
        "spatial_shapes":mx.array([[72, 72]]),
        "valid_ratios":mx.array([[[1, 1]]]),
        "tgt_mask":None,
        "memory_text":mx.random.normal((33, 1, 256)),
        "text_attention_mask":prompt_mask,
        "apply_dac": False
        
    }
    out = decoder(**inputs)
    breakpoint()



if __name__ == "__main__":
    main()
