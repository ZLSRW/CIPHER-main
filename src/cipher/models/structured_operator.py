from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..constants import SOURCE_NAMES
from .csst import ConsensusSpectralSourceTransport

class StructuredOperatorStep(nn.Module):
    """Transport, disease-conditioned source fusion, and a structured field update."""
    def __init__(self, field_dim: int, dropout: float) -> None:
        super().__init__()
        self.csst = ConsensusSpectralSourceTransport(field_dim, dropout)
        self.g_pi = nn.Sequential(
            nn.Linear(field_dim + 1 + 6 + 4, field_dim),
            nn.GELU(),
            nn.Linear(field_dim, len(SOURCE_NAMES)),
        )
        self.phi_r = nn.Sequential(
            nn.Linear(field_dim + 1 + 6, field_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(field_dim, field_dim),
        )
        self.phi_d = nn.Sequential(
            nn.Linear(1 + 6 + 1, field_dim),
            nn.GELU(),
            nn.Linear(field_dim, field_dim),
        )
        self.gate_dropout = nn.Dropout(min(0.25, dropout + 0.05))
        self.layer_norm = nn.LayerNorm(field_dim)
        self.temperature = 1.35

    def forward(
        self,
        node_field: torch.Tensor,
        source_messages: dict[str, torch.Tensor],
        data,
        alpha_D: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> dict[str, object]:
        transport_result = self.csst(node_field, source_messages, data)
        source_messages = transport_result['source_messages']
        gate_input = torch.cat(
            [node_field, data.asd_prior, data.pathway_memberships, data.source_coverage], dim=-1
        )
        source_weights = torch.softmax(
            self.g_pi(gate_input) / self.temperature, dim=-1
        )
        source_weights = self.gate_dropout(source_weights)
        source_weights = source_weights / source_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        source_stack = torch.stack([source_messages[source] for source in SOURCE_NAMES], dim=1)
        source_mixture = torch.sum(source_weights.unsqueeze(-1) * source_stack, dim=1)
        reaction_field = self.phi_r(
            torch.cat([node_field, data.asd_prior, data.pathway_memberships], dim=-1)
        )
        biological_drive = self.phi_d(
            torch.cat([data.asd_prior, data.pathway_memberships, data.module_count], dim=-1)
        )
        laplacian_field = torch.sparse.mm(data.graph_laplacian, node_field)
        structured_field = -torch.abs(alpha_D) * laplacian_field + source_mixture + reaction_field + biological_drive
        updated_field = self.layer_norm(node_field + torch.abs(delta_t) * F.gelu(structured_field))
        field_alignment_residual = (
            (updated_field - node_field) / (torch.abs(delta_t) + 1e-8) - structured_field
        )
        return {
            'node_field': updated_field,
            'source_messages': source_messages,
            'source_weights': source_weights,
            'field_alignment_residual': field_alignment_residual,
            'reaction_field': reaction_field,
            'biological_drive': biological_drive,
            'laplacian_field': laplacian_field,
            'source_mixture': source_mixture,
            'transport_result': transport_result,
        }


class ASDConditionedStructuredOperator(nn.Module):
    """Carry transported source states through the prescribed operator steps."""
    def __init__(
        self, field_dim: int, num_steps: int, dropout: float
    ) -> None:
        super().__init__()
        self.steps = nn.ModuleList(
            [
                StructuredOperatorStep(field_dim, dropout)
                for _ in range(num_steps)
            ]
        )
        self.alpha_D = nn.Parameter(torch.tensor(0.7))
        self.delta_t = nn.Parameter(torch.tensor(0.6))

    def forward(
        self, node_field: torch.Tensor, source_messages: dict[str, torch.Tensor], data
    ) -> dict[str, object]:
        field_alignment_residuals = []
        operator_source_weights = []
        transport_penalties = []
        transport_reliabilities = []
        last = {}
        for step in self.steps:
            last = step(
                node_field,
                source_messages,
                data,
                self.alpha_D,
                self.delta_t,
            )
            node_field = last['node_field']
            source_messages = last['source_messages']
            field_alignment_residuals.append(last['field_alignment_residual'])
            operator_source_weights.append(last['source_weights'])
            transport_penalties.append(last['transport_result']['transport_penalty'])
            transport_reliabilities.append(last['transport_result']['source_reliability'])
        return {
            'node_field': node_field,
            'source_messages': source_messages,
            'field_alignment_residuals': field_alignment_residuals,
            'source_weights': operator_source_weights,
            'transport_penalties': transport_penalties,
            "transport_reliabilities": transport_reliabilities,
            'reaction_field': last['reaction_field'],
            'biological_drive': last['biological_drive'],
            'laplacian_field': last['laplacian_field'],
            'source_mixture': last['source_mixture'],
        }
