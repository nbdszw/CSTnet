import json
import os
from typing import Optional

import numpy as np
import torch


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + eps)


def load_semantic_embeddings(
    semantic_path: Optional[str],
    num_class: int,
    normalize: bool = True,
) -> torch.Tensor:
    """Load class-level semantic embeddings.

    Supported file formats:
    - .npy: ndarray with shape [num_class, d_sem]
    - .pt/.pth: tensor or dict containing key 'embeddings'
    - .json: list[list[float]] or {'embeddings': list[list[float]]}

    If semantic_path is empty or missing, fallback to one-hot semantic priors.
    """
    if not semantic_path or not os.path.exists(semantic_path):
        embeddings = torch.eye(num_class, dtype=torch.float32)
        return _l2_normalize(embeddings) if normalize else embeddings

    ext = os.path.splitext(semantic_path)[1].lower()
    if ext == ".npy":
        arr = np.load(semantic_path)
        embeddings = torch.tensor(arr, dtype=torch.float32)
    elif ext in {".pt", ".pth"}:
        obj = torch.load(semantic_path, map_location="cpu")
        if isinstance(obj, dict):
            if "embeddings" in obj:
                embeddings = obj["embeddings"].float()
            else:
                raise ValueError("Semantic .pt/.pth dict must contain key 'embeddings'.")
        elif isinstance(obj, torch.Tensor):
            embeddings = obj.float()
        else:
            raise ValueError("Unsupported semantic tensor object in .pt/.pth.")
    elif ext == ".json":
        with open(semantic_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            obj = obj.get("embeddings", None)
        if obj is None:
            raise ValueError("Semantic .json must be list or contain key 'embeddings'.")
        embeddings = torch.tensor(obj, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported semantic embedding file extension: {ext}")

    if embeddings.ndim != 2:
        raise ValueError(f"Semantic embeddings must be 2D, got shape {tuple(embeddings.shape)}")
    if embeddings.shape[0] != num_class:
        raise ValueError(
            f"Semantic embeddings class count mismatch: expected {num_class}, got {embeddings.shape[0]}"
        )

    if normalize:
        embeddings = _l2_normalize(embeddings)
    return embeddings
