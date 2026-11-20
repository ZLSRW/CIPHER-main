from __future__ import annotations

import torch
import torch.nn as nn
from ..constants import SOURCE_NAMES

class ConsensusSpectralSourceTransport(nn.Module):
    """Transport source messages toward a smooth, reliability-weighted consensus."""

    def __init__(self, field_dim: int, dropout: float) -> None:
        super().__init__()
        self.g_rho = nn.Sequential(
            nn.Linear(field_dim * 3 + 1 + 6 + 4, field_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(field_dim, 1),
        )
        self.g_delta = nn.Sequential(
            nn.Linear(field_dim * 3 + 1 + 6, field_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(field_dim, field_dim),
        )
        self.g_eta = nn.Sequential(
            nn.Linear(field_dim * 2 + 1 + 6, field_dim),
            nn.GELU(),
            nn.Linear(field_dim, 1),
        )
        self.g_kappa = nn.Sequential(
            nn.Linear(field_dim * 3 + 1 + 6 + 4, field_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(field_dim, 1),
        )
        self.layer_norm = nn.LayerNorm(field_dim)
        self.beta = nn.Parameter(torch.tensor(0.35))
        self.tau = nn.Parameter(torch.tensor(0.45))

    def forward(
        self, node_field: torch.Tensor, source_messages: dict[str, torch.Tensor], data
    ) -> dict[str, object]:
        source_stack = torch.stack([source_messages[source] for source in SOURCE_NAMES], dim=1)
        source_mean = source_stack.mean(dim=1, keepdim=True)
        field_context = node_field.unsqueeze(1).expand_as(source_stack)
        asd_context = data.asd_prior.unsqueeze(1).expand(-1, len(SOURCE_NAMES), -1)
        pathway_context = data.pathway_memberships.unsqueeze(1).expand(
            -1, len(SOURCE_NAMES), -1
        )
        coverage_context = data.source_coverage.unsqueeze(1).expand(
            -1, len(SOURCE_NAMES), -1
        )
        reliability_input = torch.cat(
            [
                source_stack,
                torch.abs(source_stack - source_mean),
                field_context,
                asd_context,
                pathway_context,
                coverage_context,
            ],
            dim=-1,
        )
        source_reliability = torch.softmax(
            self.g_rho(reliability_input).squeeze(-1), dim=1
        )
        consensus_field = torch.sum(source_reliability.unsqueeze(-1) * source_stack, dim=1)
        smoothed_consensus = consensus_field - torch.sigmoid(self.beta) * torch.sparse.mm(
            data.graph_laplacian, consensus_field
        )
        consensus_gate = torch.sigmoid(
            self.g_eta(
                torch.cat(
                    [node_field, smoothed_consensus, data.asd_prior, data.pathway_memberships], dim=-1
                )
            )
        )
        consensus_field = (
            consensus_gate * smoothed_consensus
            + (1.0 - consensus_gate) * consensus_field
        )
        consensus_expanded = consensus_field.unsqueeze(1).expand_as(source_stack)
        source_correction = self.g_delta(
            torch.cat(
                [
                    source_stack,
                    consensus_expanded,
                    field_context,
                    asd_context,
                    pathway_context,
                ],
                dim=-1,
            )
        )
        retention_gate = 0.25 + 0.75 * torch.sigmoid(
            self.g_kappa(
                torch.cat(
                    [
                        source_stack,
                        consensus_expanded,
                        field_context,
                        asd_context,
                        pathway_context,
                        coverage_context,
                    ],
                    dim=-1,
                )
            )
        )
        transported_stack = self.layer_norm(
            consensus_expanded
            + retention_gate * (source_stack - consensus_expanded)
            + torch.sigmoid(self.tau) * source_correction
        )
        source_messages = {
            source: transported_stack[:, index]
            for index, source in enumerate(SOURCE_NAMES)
        }
        transported_disagreement = torch.sum(
            source_reliability.unsqueeze(-1)
            * (transported_stack - transported_stack.mean(dim=1, keepdim=True)).square(),
            dim=1,
        )
        correction_energy = source_correction.square().mean()
        smoothness = torch.sparse.mm(data.graph_laplacian, consensus_field).square().mean()
        transport_penalty = (
            transported_disagreement.mean()
            + 0.05 * correction_energy
            + 0.05 * smoothness
        )
        return {
            'source_messages': source_messages,
            'consensus_field': consensus_field,
            'source_reliability': source_reliability,
            'transport_penalty': transport_penalty,
        }
