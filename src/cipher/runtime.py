from __future__ import annotations

import random
from pathlib import Path
import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_pairs(pairs: torch.Tensor, num_nodes: int) -> None:
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype != torch.long:
        raise ValueError("pairs must be an int64 tensor of shape [N, 2]")
    if pairs.numel() and (pairs.min() < 0 or pairs.max() >= num_nodes):
        raise ValueError("Pair node indices are outside the graph")
