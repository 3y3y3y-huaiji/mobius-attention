r"""
Mobius-Attention Package
Data-Dependent Continuous Semantic Gauge Möbius Attention (SG-Möbius-RoPE)
Dual-Licensed under Apache-2.0 and MulanPSL-2.0.
"""

from .gauge_predictor import (
    HysteresisSemanticGaugePredictor,
    SemanticGaugeConnection,
    causal_viscous_ema,
)
from .mobius_attention import (
    ContinuousMobiusRoPEModulator,
    SemanticGaugeMobiusRoPE,
    SGMobiusRotaryEmbedding,
    apply_rotary_angle_addition,
    clear_model_holonomy_phase,
    inject_mobius_attention,
    nonnegative_feature_map,
    set_model_holonomy_phase,
)

__all__ = [
    "SGMobiusRotaryEmbedding",
    "ContinuousMobiusRoPEModulator",
    "SemanticGaugeMobiusRoPE",
    "SemanticGaugeConnection",
    "HysteresisSemanticGaugePredictor",
    "causal_viscous_ema",
    "nonnegative_feature_map",
    "apply_rotary_angle_addition",
    "inject_mobius_attention",
    "set_model_holonomy_phase",
    "clear_model_holonomy_phase",
]
