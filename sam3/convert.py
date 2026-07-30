import torch
import argparse
import json
from pathlib import Path
import shutil
from typing import Dict, Union, Optional

import mlx.core as mx
from huggingface_hub import snapshot_download


MLX_COMMUNITY_REPO = "mlx-community/sam3-image"
PYTORCH_REPO = "facebook/sam3"


def load_from_hub(
    hf_repo: str = MLX_COMMUNITY_REPO,
    local_dir: Optional[str] = None,
) -> Path:
    download_kwargs = {
        "repo_id": hf_repo,
        "allow_patterns": ["*.safetensors", "*.json"],
    }
    
    if local_dir:
        download_kwargs["local_dir"] = local_dir
    
    model_path = Path(snapshot_download(**download_kwargs))
    weights_file = model_path / "model.safetensors"
    
    if not weights_file.exists():
        raise FileNotFoundError(f"model.safetensors not found in {hf_repo}.")
    
    return weights_file


def save_weights(save_path: Union[str, Path], weights: Dict[str, mx.array]) -> None:
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    total_size = sum(v.nbytes for v in weights.values())
    index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}

    model_path = save_path / "model.safetensors"
    mx.save_safetensors(str(model_path), weights)
    
    for weight_name in weights.keys():
        index_data["weight_map"][weight_name] = "model.safetensors"
    
    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }

    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(index_data, f, indent=4)

def download(hf_repo):
    return Path(
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=["*.pt", "*.json"],
        )
    )

def update_attn_keys(key, mlx_weights):
    value = mlx_weights[key]
    del mlx_weights[key]
    
    if "in_proj_weight" in key:
        qkv, _ = value.shape[0], value.shape[1]
        qkv_dim = qkv // 3
        key_prefix = key.rsplit('.', 1)[0]
        new_dict = {
            f"{key_prefix}.query_proj.weight": value[0:qkv_dim, :],
            f"{key_prefix}.key_proj.weight": value[qkv_dim:2*qkv_dim, :],
            f"{key_prefix}.value_proj.weight": value[2*qkv_dim: , :],
        }
        mlx_weights.update(new_dict)
    
    if "in_proj_bias" in key:
        qkv = value.shape[0]
        qkv_dim = qkv // 3
        key_prefix = key.rsplit('.', 1)[0]
        new_dict = {
            f"{key_prefix}.query_proj.bias": value[0:qkv_dim],
            f"{key_prefix}.key_proj.bias": value[qkv_dim:2*qkv_dim],
            f"{key_prefix}.value_proj.bias": value[2*qkv_dim: ],
        }
        mlx_weights.update(new_dict)
        
# vision and language backbone, transformer fusion encoder / detr decoder,
# dot product scoring mlp layer, segmentation head, geometry encoder
CONVERTED_PREFIXES = (
    "backbone.",
    "transformer.",
    "dot_prod_scoring.",
    "segmentation_head.",
    "geometry_encoder.",
)


def model_layout():
    """Conv permutations and parameter shapes of the MLX model, keyed by parameter name.

    Every optional module is enabled: the published weights have to serve every
    configuration, not just the one a particular caller happens to build.
    """
    # Imported here because model_builder imports this module at module level.
    from sam3.model_builder import (
        build_sam3_image_model_skeleton,
        conv_weight_perms,
        parameter_shapes,
    )

    model = build_sam3_image_model_skeleton(
        enable_segmentation=True,
        enable_inst_interactivity=True,
    )
    return conv_weight_perms(model), parameter_shapes(model)


def validate_layout(mlx_weights, expected):
    mismatched = {
        k: (tuple(v.shape), expected[k])
        for k, v in mlx_weights.items()
        if k in expected and tuple(v.shape) != expected[k]
    }
    if mismatched:
        details = "\n".join(
            f"  {k}: converted {got}, model expects {exp}"
            for k, (got, exp) in sorted(mismatched.items())
        )
        raise ValueError(
            f"{len(mismatched)} converted weight(s) do not match the MLX model:\n{details}"
        )

    unknown = sorted(k for k in mlx_weights if k not in expected)
    if unknown:
        print(f"Warning: {len(unknown)} converted key(s) are not model parameters: {unknown}")


def convert(model_path):
    weight_file = str(model_path / "sam3.pt")
    weights = torch.load(weight_file, map_location="cpu", weights_only=True)

    perms, expected = model_layout()

    mlx_weights = dict()
    for k, v in weights.items():
        if "detector" not in k:
            continue

        k = k.replace("detector.", "")
        if not k.startswith(CONVERTED_PREFIXES):
            continue

        v = mx.array(v.numpy())
        perm = perms.get(k)
        if perm is not None and v.ndim == 4 and tuple(v.shape) != expected.get(k):
            v = v.transpose(*perm)
        mlx_weights[k] = v

        if k.endswith("in_proj_weight") or k.endswith("in_proj_bias"):
            update_attn_keys(k, mlx_weights)

    validate_layout(mlx_weights, expected)

    return mlx_weights

def download_and_convert(
    hf_repo: str = PYTORCH_REPO,
    mlx_path: Union[str, Path] = "sam3-mod-weights",
    force: bool = False
) -> Path:
    mlx_path = Path(mlx_path)
    weights_file = mlx_path / "model.safetensors"
    index_file = mlx_path / "model.safetensors.index.json"
    
    if weights_file.exists() and index_file.exists() and not force:
        return weights_file
    
    print(f"Downloading and converting weights from {hf_repo}...")
    model_path = download(hf_repo)

    mlx_path.mkdir(parents=True, exist_ok=True)
    
    mlx_weights = convert(model_path)
    save_weights(mlx_path, mlx_weights)

    return weights_file
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SAM-3 MLX weights or convert from PyTorch")
    parser.add_argument(
        "--mlx-repo",
        default=MLX_COMMUNITY_REPO,
        type=str,
        help=f"MLX Community repo to download pre-converted weights (default: {MLX_COMMUNITY_REPO})",
    )
    parser.add_argument(
        "--pytorch-repo",
        default=PYTORCH_REPO,
        type=str,
        help=f"PyTorch repo to download and convert weights (default: {PYTORCH_REPO})",
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        default=None,
        help="Local path to save/cache the MLX Model weights."
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert from PyTorch weights instead of loading pre-converted MLX weights"
    )
    args = parser.parse_args()

    if args.convert:
        mlx_path = args.mlx_path or "sam3-mod-weights"
        print(f"Converting PyTorch weights from {args.pytorch_repo}...")
        model_path = download(args.pytorch_repo)
        
        mlx_path = Path(mlx_path)
        mlx_path.mkdir(parents=True, exist_ok=True)
        
        mlx_weights = convert(model_path)
        save_weights(mlx_path, mlx_weights)
        print(f"Converted weights saved to {mlx_path}")
    else:
        print(f"Downloading MLX weights from {args.mlx_repo}...")
        weights_path = load_from_hub(args.mlx_repo, args.mlx_path)
        print(f"MLX weights available at: {weights_path}")