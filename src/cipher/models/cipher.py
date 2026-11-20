from __future__ import annotations

import torch
import torch.nn as nn
from ..config import CIPHERConfig
from .source_encoder import SourceSpecificEncoder
from .structured_operator import ASDConditionedStructuredOperator
from .relational_decoder import SourceFactorTokenizer, SourceFactorRelationalDecoder

class CIPHER(nn.Module):
    """Consensus-informed propagation for multi-source graph-knowledge fusion."""
    def __init__(self, config: CIPHERConfig) -> None:
        super().__init__()
        self.config = config
        self.phi_x = nn.Sequential(
            nn.Linear(config.node_feature_dim, config.field_dim),
            nn.LayerNorm(config.field_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.phi_t = nn.Sequential(
            nn.Linear(4, config.field_dim),
            nn.LayerNorm(config.field_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.local_encoder = SourceSpecificEncoder(
            config.field_dim,
            config.chebyshev_order,
            config.num_local_layers,
            config.dropout,
        )
        self.structured_operator = ASDConditionedStructuredOperator(
            config.field_dim, config.num_operator_steps, config.dropout
        )
        self.tokenizer = SourceFactorTokenizer(
            config.field_dim,
            config.token_dim,
            config.num_pathway_factors,
            config.dropout,
        )
        self.relational_decoder = SourceFactorRelationalDecoder(config.token_dim, config.dropout)

    def encode(self, data) -> dict[str, object]:
        initial_field = self.phi_x(data.node_features) + self.phi_t(data.topology_features)
        local = self.local_encoder(initial_field, data)
        structured_operator = self.structured_operator(local['node_field'], local['source_messages'], data)
        node_field = structured_operator['node_field']
        source_messages = structured_operator['source_messages']
        field_alignment_residuals = structured_operator['field_alignment_residuals']
        operator_source_weights = structured_operator['source_weights']
        transport_penalties = structured_operator['transport_penalties']
        transport_reliabilities = structured_operator["transport_reliabilities"]
        source_factor_tokens = self.tokenizer(source_messages, node_field, data)
        return {
            'initial_field': initial_field,
            'encoded_field': local['node_field'],
            'final_field': node_field,
            'local_source_weights': local['source_weights'],
            'operator_source_weights': operator_source_weights,
            'field_alignment_residuals': field_alignment_residuals,
            'transport_penalties': transport_penalties,
            "transport_reliabilities": transport_reliabilities,
            'source_factor_tokens': source_factor_tokens,
        }

    def forward(self, data, pairs: torch.Tensor) -> dict[str, object]:
        encoded = self.encode(data)
        output = self.relational_decoder(encoded['source_factor_tokens'], pairs)
        encoded.update(output)
        return encoded
