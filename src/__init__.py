r"""
Mobius-Attention Package
Data-Dependent Continuous Semantic Gauge Möbius Attention (SG-Möbius-RoPE)
Dual-Licensed under Apache-2.0 and MulanPSL-2.0.
"""

from .mobius_attention import (
    ContinuousMobiusRoPEModulator,
    SemanticGaugeConnection,
    SemanticGaugeMobiusRoPE,
    apply_rotary_angle_addition,
    causal_viscous_ema,
    clear_model_holonomy_phase,
    inject_mobius_attention,
    nonnegative_feature_map,
    set_model_holonomy_phase,
)

__all__ = [
    "SemanticGaugeConnection",
    "ContinuousMobiusRoPEModulator",
    "SemanticGaugeMobiusRoPE",
    "inject_mobius_attention",
    "causal_viscous_ema",
    "nonnegative_feature_map",
    "apply_rotary_angle_addition",
    "set_model_holonomy_phase",
    "clear_model_holonomy_phase"
]
