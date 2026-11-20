from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class CIPHERConfig:
    """Representation widths and propagation depths for CIPHER."""
    node_feature_dim: int = 64
    field_dim: int = 224
    token_dim: int = 56
    chebyshev_order: int = 4
    num_local_layers: int = 2
    num_operator_steps: int = 3
    num_pathway_factors: int = 6
    dropout: float = 0.10

    def __post_init__(self) -> None:
        if self.num_pathway_factors != 6:
            raise ValueError("CIPHER requires six pathway factors")
        if self.token_dim <= 0 or self.token_dim % 4:
            raise ValueError("token_dim must be positive and divisible by four")
        if min(self.node_feature_dim, self.field_dim, self.chebyshev_order,
               self.num_local_layers, self.num_operator_steps) < 1:
            raise ValueError("Model dimensions and depths must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

