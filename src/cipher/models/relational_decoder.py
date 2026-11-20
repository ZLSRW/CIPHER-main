from __future__ import annotations

import torch
import torch.nn as nn
from ..constants import SOURCE_NAMES

class SourceFactorTokenizer(nn.Module):
    """Seven tokens per source: one generic token and six pathway factors."""
    def __init__(
        self, field_dim: int, token_dim: int, num_pathway_factors: int, dropout: float
    ) -> None:
        super().__init__()
        self.num_tokens = num_pathway_factors + 1
        self.psi_source_factor = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(field_dim * 2, token_dim)
                        for _ in range(self.num_tokens)
                    ]
                )
                for _ in SOURCE_NAMES
            ]
        )
        self.source_attention = nn.ModuleList(
            [
                nn.MultiheadAttention(token_dim, num_heads=4, batch_first=True)
                for _ in SOURCE_NAMES
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        source_messages: dict[str, torch.Tensor],
        final_field: torch.Tensor,
        data,
    ) -> torch.Tensor:
        source_blocks = []
        for source_index, source in enumerate(SOURCE_NAMES):
            token_context = torch.cat([source_messages[source], final_field], dim=-1)
            source_factor_tokens = []
            for factor_index in range(self.num_tokens):
                token = self.psi_source_factor[source_index][factor_index](token_context)
                if factor_index > 0:
                    token = (
                        token
                        + data.pathway_memberships[:, factor_index - 1 : factor_index]
                        * data.asd_prior
                    )
                source_factor_tokens.append(token)
            raw_tokens = torch.stack(source_factor_tokens, dim=1)
            attended_tokens, _ = self.source_attention[source_index](
                raw_tokens, raw_tokens, raw_tokens, need_weights=False
            )
            source_blocks.append(self.dropout(raw_tokens + attended_tokens))
        return torch.stack(source_blocks, dim=1)


class SourceFactorRelationalDecoder(nn.Module):
    """Symmetric PPI scoring with classifier-consistent source-factor terms."""
    def __init__(self, token_dim: int, dropout: float) -> None:
        super().__init__()
        decoder_dim = token_dim * 4
        self.token_dim = token_dim
        self.phi_q = nn.Sequential(
            nn.Linear(token_dim * 3, decoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_dim, token_dim),
        )
        self.phi_c = nn.Sequential(
            nn.Linear(token_dim * 3, decoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_dim, token_dim),
        )
        self.W_q = nn.Linear(token_dim, token_dim)
        self.W_c = nn.Linear(token_dim, token_dim)
        self.g = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, 1),
        )
        self.g_mech = nn.Sequential(
            nn.Linear(token_dim * 5, decoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_dim, 1),
        )
        self.pair_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self, source_factor_tokens: torch.Tensor, pairs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        left_tokens = source_factor_tokens[pairs[:, 0]]
        right_tokens = source_factor_tokens[pairs[:, 1]]
        left_pooled_tokens = left_tokens.mean(dim=(1, 2))
        right_pooled_tokens = right_tokens.mean(dim=(1, 2))
        pair_context = torch.cat(
            [
                left_pooled_tokens + right_pooled_tokens,
                torch.abs(left_pooled_tokens - right_pooled_tokens),
                left_pooled_tokens * right_pooled_tokens,
            ],
            dim=-1,
        )
        pair_query = self.phi_q(pair_context)
        compatibility_vectors = self.phi_c(
            torch.cat([left_tokens + right_tokens, torch.abs(left_tokens - right_tokens), left_tokens * right_tokens], dim=-1)
        )
        compatibility_gate = torch.sigmoid(
            (
                self.W_q(pair_query).unsqueeze(1).unsqueeze(1)
                * self.W_c(compatibility_vectors)
            ).sum(dim=-1)
            / (self.token_dim**0.5)
        )
        source_factor_scores = self.g(compatibility_vectors).squeeze(-1)
        source_factor_contributions = compatibility_gate * source_factor_scores
        pair_representation = torch.cat(
            [
                left_pooled_tokens + right_pooled_tokens,
                torch.abs(left_pooled_tokens - right_pooled_tokens),
                left_pooled_tokens * right_pooled_tokens,
                0.5 * (left_pooled_tokens + right_pooled_tokens),
                pair_query,
            ],
            dim=-1,
        )
        pair_logits = (
            self.pair_bias
            + source_factor_contributions.mean(dim=(1, 2))
            + self.g_mech(pair_representation).squeeze(-1)
        )
        return {
            'pair_logits': pair_logits,
            'pair_representation': pair_representation,
            'source_factor_contributions': source_factor_contributions,
            "compatibility_gate": compatibility_gate,
        }
