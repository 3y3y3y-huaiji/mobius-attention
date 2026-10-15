r"""
Mobius-Attention Package
Data-Dependent Semantic Gauge Möbius Attention (SG-Möbius-RoPE)
Dual-Licensed under Apache-2.0 and MulanPSL-2.0.
"""

from .mobius_attention import (
    SemanticGaugeConnection,
    SemanticGaugeMobiusRoPE,
    inject_mobius_attention
)

__all__ = [
    "SemanticGaugeConnection",
    "SemanticGaugeMobiusRoPE",
    "inject_mobius_attention"
]
