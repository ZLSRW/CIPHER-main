from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from ..constants import SOURCE_NAMES

class SourceSpecificMessageLayer(nn.Module):
    """Source-specific Chebyshev messages fused through the g_alpha gate."""
    def __init__(self, field_dim: int, chebyshev_order: int, dropout: float) -> None:
        super().__init__()
        self.convolutions = nn.ModuleDict(
            {
                source: ChebConv(field_dim, field_dim, K=chebyshev_order)
                for source in SOURCE_NAMES
            }
        )
        self.g_alpha = nn.Sequential(
            nn.Linear(field_dim + 2, field_dim),
            nn.GELU(),
            nn.Linear(field_dim, len(SOURCE_NAMES)),
        )
        self.gate_dropout = nn.Dropout(min(0.25, dropout + 0.05))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(field_dim)
        self.temperature = 1.4

    def forward(
        self, node_field: torch.Tensor, data
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        source_messages = {
            source: self.convolutions[source](
                node_field, data.source_edge_indices[source]
            )
            for source in SOURCE_NAMES
        }
        source_stack = torch.stack([source_messages[source] for source in SOURCE_NAMES], dim=1)
        gate_context = torch.cat([node_field, data.asd_prior, data.module_count], dim=-1)
        source_weights = torch.softmax(self.g_alpha(gate_context) / self.temperature, dim=-1)
        source_weights = self.gate_dropout(source_weights)
        source_weights = source_weights / source_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        fused_message = torch.sum(source_weights.unsqueeze(-1) * source_stack, dim=1)
        updated_field = self.layer_norm(node_field + self.dropout(F.gelu(fused_message)))
        return updated_field, source_weights, source_messages


class SourceSpecificEncoder(nn.Module):
    """Initialize the node field and preserve the final source-specific messages."""
    def __init__(
        self, field_dim: int, chebyshev_order: int, num_layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SourceSpecificMessageLayer(field_dim, chebyshev_order, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, node_field: torch.Tensor, data) -> dict[str, object]:
        local_source_weights = []
        source_messages = {}
        for layer in self.layers:
            node_field, source_weights, source_messages = layer(node_field, data)
            local_source_weights.append(source_weights)
        return {'node_field': node_field, 'source_messages': source_messages, 'source_weights': local_source_weights}
