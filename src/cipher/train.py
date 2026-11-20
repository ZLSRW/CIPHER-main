"""Train CIPHER on one fixed validation fold."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .config import CIPHERConfig
from .data import CIPHERDataset
from .losses import LossConfig, cipher_objective
from .metrics import f1_threshold, summarize
from .models import CIPHER
from .runtime import load_config, set_seed


def train_fold(dataset_dir: str | Path, fold: int, config: dict,
               output_dir: str | Path, device_name: str = "cpu") -> dict:
    training = config["training"]
    if min(int(training["epochs"]), int(training["patience"])) < 1:
        raise ValueError("epochs and patience must be positive")
    seed = int(training["seed"])
    set_seed(seed)
    device = torch.device(device_name)
    dataset = CIPHERDataset(dataset_dir, fold)
    graph = dataset.data.to(device)
    splits = dataset.splits
    train_pairs = splits.train_pairs.to(device)
    train_labels = splits.train_labels.to(device)
    validation_pairs = splits.validation_pairs.to(device)
    validation_labels = splits.validation_labels.numpy()
    model_config = CIPHERConfig(**config["model"])
    if model_config.node_feature_dim != graph.node_features.shape[1]:
        raise ValueError("Configured node_feature_dim differs from the dataset")
    loss_config = LossConfig(**config["loss"])
    model = CIPHER(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
        lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    best_auc, best_epoch, best_state = -np.inf, 0, None
    patience_left = int(training["patience"])
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(graph, train_pairs)
        losses = cipher_objective(output, train_labels, epoch, loss_config)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
        optimizer.step()
        del output, losses
        model.eval()
        with torch.no_grad():
            probabilities = model(graph, validation_pairs)["pair_logits"].sigmoid().cpu().numpy()
        validation_auc = float(roc_auc_score(validation_labels, probabilities))
        print(f"epoch={epoch} validation_auc={validation_auc:.6f}", flush=True)
        if validation_auc > best_auc:
            best_auc, best_epoch = validation_auc, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            patience_left = int(training["patience"])
        else:
            patience_left -= 1
            if patience_left == 0:
                break
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint was produced")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = model(graph, validation_pairs)["pair_logits"].sigmoid().cpu().numpy()
    threshold = f1_threshold(validation_labels, probabilities)
    checkpoint = {"format_version": 1, "model_state": best_state,
        "model_config": asdict(model_config), "loss_config": asdict(loss_config),
        "training_config": training, "seed": seed, "fold": fold,
        "dataset": dataset.metadata.get("dataset_name", Path(dataset_dir).name),
        "best_epoch": best_epoch, "validation_auc": best_auc, "threshold": threshold}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_dir / "cipher.pt")
    metrics = summarize(validation_labels, probabilities, threshold)
    (output_dir / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--config", default="configs/cipher.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(json.dumps(train_fold(args.dataset_dir, args.fold, load_config(args.config),
                                args.output_dir, args.device), indent=2))


if __name__ == "__main__":
    main()
