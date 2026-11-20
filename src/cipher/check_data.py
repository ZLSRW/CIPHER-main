"""Check graph/supervision exclusion and the supplied fixed split files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def pair_keys(frame: pd.DataFrame, num_nodes: int) -> np.ndarray:
    values = frame[["left", "right"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError("Pair indices must be finite integers")
    pairs = values.astype(np.int64)
    if len(pairs) and (pairs.min() < 0 or pairs.max() >= num_nodes):
        raise ValueError("Invalid node index")
    if (pairs[:, 0] == pairs[:, 1]).any():
        raise ValueError("Self interactions are not supported")
    return np.minimum(pairs[:, 0], pairs[:, 1]) * num_nodes + np.maximum(pairs[:, 0], pairs[:, 1])


def audit_dataset(dataset_dir: str | Path) -> dict:
    directory = Path(dataset_dir)
    nodes = pd.read_csv(directory / "PPI_node.csv", usecols=["node_id"])
    n = len(nodes)
    if not np.array_equal(nodes["node_id"], np.arange(n)):
        raise ValueError("node_id must be contiguous and ordered")
    graph = pd.read_csv(directory / "PPI_edge.csv")
    graph_keys = pair_keys(graph, n)
    source_keys = pair_keys(pd.read_csv(directory / "edge_source_flags.csv"), n)
    if not np.array_equal(np.sort(graph_keys), np.sort(source_keys)):
        raise ValueError("Source flags must index the same pruned union graph")

    def load(path):
        frame = pd.read_csv(path)
        if not frame["label"].isin([0, 1]).all():
            raise ValueError("Labels must be binary")
        keys = pair_keys(frame, n)
        if len(keys) != len(np.unique(keys)):
            raise ValueError("Repeated unordered pair in a split")
        return dict(zip(keys.tolist(), frame["label"].astype(int).tolist()))

    test = load(directory / "split_test.csv")
    reference = None
    validation_keys = set()
    folds = []
    for fold in range(1, 6):
        train = load(directory / "splits" / f"fold_{fold}" / "train.csv")
        validation = load(directory / "splits" / f"fold_{fold}" / "val.csv")
        if set(train) & set(validation) or set(test) & (set(train) | set(validation)):
            raise ValueError("Split overlap detected")
        development = dict(train)
        development.update(validation)
        if reference is None:
            reference = development
        elif reference != development:
            raise ValueError("Development labels or pair membership changed across folds")
        if validation_keys & set(validation):
            raise ValueError("Validation folds overlap")
        validation_keys.update(validation)
        folds.append({"fold": fold, "train_pairs": len(train), "validation_pairs": len(validation)})
    if validation_keys != set(reference):
        raise ValueError("Validation folds do not partition the development set")
    supervision = np.array(list(reference) + list(test), dtype=np.int64)
    if np.intersect1d(graph_keys, supervision).size:
        raise ValueError("Supervised pair remains in the propagation graph")
    return {"nodes": n, "undirected_graph_edges": len(graph_keys), "independent_test_pairs": len(test),
            "supervision_pairs": len(supervision), "folds": folds, "target_pairs_excluded": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(audit_dataset(args.dataset_dir), indent=2))


if __name__ == "__main__":
    main()
