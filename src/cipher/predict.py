"""Score indexed or symbol-mapped pairs on a fixed graph using a CIPHER checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from .config import CIPHERConfig
from .data import CIPHERDataset
from .models import CIPHER
from .runtime import validate_pairs


def predict_pairs(model, graph, pairs: torch.Tensor, batch_size: int = 512) -> dict[str, np.ndarray]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    validate_pairs(pairs, graph.num_nodes)
    if not len(pairs):
        raise ValueError("Provide at least one pair")
    model.eval()
    device = graph.node_features.device
    values = {key: [] for key in ("pair_logits", "pair_representation", "source_factor_contributions")}
    with torch.no_grad():
        # Encode once; batching applies to pairs, not graph neighborhoods.
        encoded = model.encode(graph)
        for block in pairs.split(batch_size):
            decoded = model.relational_decoder(encoded["source_factor_tokens"], block.to(device))
            for key in values:
                values[key].append(decoded[key].cpu().numpy())
        del encoded
    result = {key: np.concatenate(parts, axis=0) for key, parts in values.items()}
    result["probability"] = torch.from_numpy(result["pair_logits"]).sigmoid().numpy()
    return result


def load_checkpoint(path: str | Path, device: torch.device):
    # Checkpoints contain tensors and plain metadata, with no custom pickle classes.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format; use weights created by this package")
    model = CIPHER(CIPHERConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pairs", required=True, help="CSV with left,right columns")
    parser.add_argument("--pair-format", choices=("indices", "symbols"), default="indices")
    parser.add_argument("--output", required=True)
    parser.add_argument("--representations", help="Optional output NPZ for representations and source-factor terms")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    dataset = CIPHERDataset(args.dataset_dir, fold=None)
    dataset_name = dataset.metadata.get("dataset_name", Path(args.dataset_dir).name)
    if checkpoint["dataset"] != dataset_name:
        raise ValueError("Checkpoint and dataset differ; use the graph used for training")
    graph = dataset.data.to(device)
    frame = pd.read_csv(args.pairs)
    if args.pair_format == "symbols":
        symbols = dataset.node_df["symbol"].astype(str)
        if symbols.duplicated().any():
            raise ValueError("Dataset symbols are ambiguous; use node indices")
        mapping = dict(zip(symbols, dataset.node_df["node_id"]))
        indexed = frame[["left", "right"]].astype(str).apply(lambda col: col.map(mapping))
        if indexed.isna().any().any():
            raise ValueError("Unrecognized protein symbol; use dataset identifiers exactly")
    else:
        indexed = frame[["left", "right"]].apply(pd.to_numeric, errors="raise")
        array = indexed.to_numpy(dtype=np.float64)
        if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
            raise ValueError("Node indices must be finite integers")
    pairs = torch.as_tensor(indexed.to_numpy(dtype=np.int64))
    values = predict_pairs(model, graph, pairs, args.batch_size)
    frame["probability"] = values["probability"]
    frame["logit"] = values["pair_logits"]
    frame["prediction"] = (values["probability"] >= checkpoint["threshold"]).astype(int)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    if args.representations:
        artifact = Path(args.representations)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(values, pairs=pairs.numpy())
        if "label" in frame:
            arrays["labels"] = frame["label"].to_numpy()
        np.savez_compressed(artifact, **arrays)
    print(f"Scored {len(frame)} pairs")


if __name__ == "__main__":
    main()
