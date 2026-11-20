from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch_geometric.data import Data

from .constants import MODULE_COLUMNS, SOURCE_NAMES


REQUIRED_FILES = (
    "PPI_node.csv",
    "PPI_edge.csv",
    "topology_features.csv",
    "sfari_node_prior.csv",
    "reactome_module_matrix.csv",
    "edge_source_flags.csv",
    "node_source_coverage.csv",
    "split_test.csv",
    "metadata.json",
)


@dataclass(frozen=True)
class SplitEdges:
    train_pairs: torch.Tensor
    train_labels: torch.Tensor
    validation_pairs: torch.Tensor
    validation_labels: torch.Tensor
    test_pairs: torch.Tensor
    test_labels: torch.Tensor


def _parse_feature(value: str) -> np.ndarray:
    return np.fromstring(str(value), sep=",", dtype=np.float32)


class CIPHERDataset:
    """Load one fixed leakage-controlled CIPHER benchmark and fold."""

    def __init__(self, dataset_dir: str | Path, fold: int | None = 1) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.fold = None if fold is None else int(fold)
        if self.fold is not None and self.fold not in range(1, 6):
            raise ValueError("fold must be 1 through 5, or None for graph-only loading")
        self._validate_layout()
        self.node_df = pd.read_csv(self.dataset_dir / "PPI_node.csv")
        self.edge_df = pd.read_csv(self.dataset_dir / "PPI_edge.csv")
        self.topology_df = pd.read_csv(self.dataset_dir / "topology_features.csv")
        self.prior_df = pd.read_csv(self.dataset_dir / "sfari_node_prior.csv")
        self.module_df = pd.read_csv(self.dataset_dir / "reactome_module_matrix.csv")
        self.flags_df = pd.read_csv(self.dataset_dir / "edge_source_flags.csv")
        self.coverage_df = pd.read_csv(self.dataset_dir / "node_source_coverage.csv")
        self.metadata = json.loads(
            (self.dataset_dir / "metadata.json").read_text(encoding="utf-8")
        )
        self.node_count = len(self.node_df)
        node_ids = np.arange(self.node_count)
        for table in (self.node_df, self.topology_df, self.prior_df,
                      self.module_df, self.coverage_df):
            if not np.array_equal(table["node_id"].to_numpy(), node_ids):
                raise ValueError("Node tables must have the same contiguous node_id order")
        self.data = self._build_graph()
        self.splits = None if self.fold is None else self._build_splits()

    def _validate_layout(self) -> None:
        missing = [name for name in REQUIRED_FILES if not (self.dataset_dir / name).is_file()]
        fold_dir = self.dataset_dir / "splits" / f"fold_{self.fold}"
        missing.extend(
            str(Path("splits") / f"fold_{self.fold}" / name)
            for name in ("train.csv", "val.csv")
            if self.fold is not None and not (fold_dir / name).is_file()
        )
        if missing:
            raise FileNotFoundError(
                f"Dataset {self.dataset_dir} is incomplete; missing: {', '.join(missing)}"
            )

    @staticmethod
    def _undirected(edge_pairs: np.ndarray) -> torch.Tensor:
        pairs = torch.tensor(edge_pairs.T, dtype=torch.long)
        return torch.cat([pairs, pairs.flip(0)], dim=1)

    def _laplacian(self, pairs: torch.Tensor) -> torch.Tensor:
        row, col = pairs.cpu().numpy()
        values = np.ones(row.shape[0], dtype=np.float32)
        adjacency = sparse.coo_matrix(
            (values, (row, col)), shape=(self.node_count, self.node_count)
        ).tocsr()
        degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
        degree[degree == 0] = 1.0
        inv_sqrt = sparse.diags(np.power(degree, -0.5))
        graph_laplacian = (sparse.eye(self.node_count, dtype=np.float32) - inv_sqrt @ adjacency @ inv_sqrt).tocoo()
        indices = torch.tensor(np.vstack([graph_laplacian.row, graph_laplacian.col]), dtype=torch.long)
        values = torch.tensor(graph_laplacian.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(
            indices, values, size=(self.node_count, self.node_count)
        ).coalesce()

    def _build_graph(self) -> Data:
        feature_columns = [column for column in self.node_df if column.startswith("feature_")]
        if len(feature_columns) != 1:
            raise ValueError(f"Expected one feature_* column, found {feature_columns}")
        node_features = np.vstack(self.node_df[feature_columns[0]].map(_parse_feature)).astype(np.float32)
        if not np.isfinite(node_features).all():
            raise ValueError("Node features must be finite")
        layer_norm = np.linalg.norm(node_features, axis=1, keepdims=True)
        nonzero = layer_norm.squeeze(-1) > 0
        node_features[nonzero] /= layer_norm[nonzero]

        topology_features = self.topology_df[
            ["log_degree", "pagerank", "clustering", "core_number"]
        ].to_numpy(dtype=np.float32)
        asd_prior = self.prior_df[["sfari_prior"]].to_numpy(dtype=np.float32)
        pathway_memberships = self.module_df[list(MODULE_COLUMNS)].to_numpy(dtype=np.float32)
        source_coverage = self.coverage_df[
            ["string_cov", "biogrid_cov", "intact_cov", "consensus_cov"]
        ].to_numpy(dtype=np.float32)
        pairs = self._undirected(
            self.edge_df[["left", "right"]].to_numpy(dtype=np.int64)
        )
        data = Data(
            node_features=torch.tensor(node_features),
            topology_features=torch.tensor(topology_features),
            asd_prior=torch.tensor(asd_prior),
            pathway_memberships=torch.tensor(pathway_memberships),
            module_count=torch.tensor(pathway_memberships.sum(axis=1, keepdims=True), dtype=torch.float32),
            source_coverage=torch.tensor(source_coverage),
            union_edge_index=pairs,
            graph_laplacian=self._laplacian(pairs),
            num_nodes=self.node_count,
        )
        source_edge_indices = {}
        for source in SOURCE_NAMES:
            pairs = self.flags_df.loc[
                self.flags_df[source] > 0, ["left", "right"]
            ].to_numpy(dtype=np.int64)
            source_edge_indices[source] = self._undirected(pairs)
        data.source_edge_indices = source_edge_indices
        return data

    @staticmethod
    def _load_edges(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        frame = pd.read_csv(path)
        pairs = torch.tensor(
            frame[["left", "right"]].to_numpy(dtype=np.int64), dtype=torch.long
        )
        labels = torch.tensor(frame["label"].to_numpy(dtype=np.float32))
        return pairs, labels

    def _build_splits(self) -> SplitEdges:
        fold_dir = self.dataset_dir / "splits" / f"fold_{self.fold}"
        train_pairs, train_labels = self._load_edges(fold_dir / "train.csv")
        validation_pairs, validation_labels = self._load_edges(fold_dir / "val.csv")
        test_pairs, test_labels = self._load_edges(self.dataset_dir / "split_test.csv")
        return SplitEdges(
            train_pairs=train_pairs,
            train_labels=train_labels,
            validation_pairs=validation_pairs,
            validation_labels=validation_labels,
            test_pairs=test_pairs,
            test_labels=test_labels,
        )

