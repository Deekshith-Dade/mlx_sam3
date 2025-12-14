import mlx.core as mx
import mlx.nn as nn
from sam3.model.vitdet import Attention, ViT

def _compare_arrays(self, mlx_out, i):
    import numpy as np
    prefix = "/Users/deekshith/Documents/Projects/vision-models/mlx_sam3/vit_block"
    torch_out = mx.array(np.load(f"{prefix}/state_snapshot.npy"))
    # diff = mx.abs(torch_out - mlx_out)
    # print(f"Max Diff:  {diff.max().item():.6f}")
    # print(f"Mean Diff: {diff.mean().item():.6f}")

    # Metric B: Cosine Similarity (The "Structure" Check)
    # Use this since your means/stds are drifting. 
    # If this is 0.9999+, your model is structurally perfect, just scaled differently.
    similarity = (torch_out * mlx_out).sum() / (mx.linalg.norm(torch_out) * mx.linalg.norm(mlx_out))
    print(f"Cosine Sim: {similarity.item():.6f}")
    return torch_out

def main():
    print("Hello, World!")

    params = {
        "dim": 1024,
        "num_heads": 16,
        "qkv_bias": True,
        "use_rel_pos": False,
        "rel_pos_zero_init": True,
        "input_size": (24, 24),
        "cls_token": False,
        "use_rope": True,
        "rope_theta": 10000.0,
        "rope_pt_size": (24, 24),
        "rope_interp": True,
    }
    
    attention_layer = Attention(**params)
    print("Attention layer created successfully.")
    test = mx.random.normal((9, 24, 24, 1024))
    output = attention_layer(test)
    print(output.shape)

    
    vit_params = {
        "img_size": 1008,
        "patch_size": 14,
        "in_chans": 3,
        "embed_dim": 1024,
        "depth": 32,
        "num_heads": 16,
        "mlp_ratio": 4.625,
        "qkv_bias": True,
        "drop_path_rate": 0.1,
        "norm_layer": nn.LayerNorm,
        "act_layer": nn.GELU,
        "use_abs_pos": True,
        "tile_abs_pos": True,
        "rel_pos_blocks": (),
        "rel_pos_zero_init": True,
        "window_size": 24,
        "global_att_blocks": (7, 15, 23, 31),
        "use_rope": True,
        "rope_pt_size": None,
        "use_interp_rope": True,
        "pretrain_img_size": 336,
        "pretrain_use_cls_token": True,
        "retain_cls_token": False,
        "dropout": 0.0,
        "return_interm_layers": False,
        "init_values": None,
        "ln_pre": True,
        "ln_post": False,
        "bias_patch_embed": False,
        "compile_mode": None,
        "use_act_checkpoint": False
    }
    model = ViT(**vit_params)
    x = mx.random.normal((1, 3, 1008, 1008))
    out = model(x)
    print(out[0].shape)

    breakpoint()

if __name__ == "__main__":
    main()