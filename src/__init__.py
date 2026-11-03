r"""
Mobius-Attention Package
Data-Dependent Continuous Semantic Gauge Möbius Attention (SG-Möbius-RoPE)
Dual-Licensed under Apache-2.0 and MulanPSL-2.0.
"""

from .gauge_predictor import (
    HysteresisSemanticGaugePredictor,
    SemanticGaugeConnection,
    causal_viscous_ema,
    parallel_affine_scan,
    two_stage_associative_scan,
)
from .kv_cache import SGMobiusKVCache
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
from .phase_attention import (
    apply_phase,
    attention_scores,
    causal_attention_weights,
    reversal_phase,
    rope_tables,
)

__all__ = [
    "SGMobiusRotaryEmbedding",
    "ContinuousMobiusRoPEModulator",
    "SemanticGaugeMobiusRoPE",
    "SemanticGaugeConnection",
    "HysteresisSemanticGaugePredictor",
    "causal_viscous_ema",
    "parallel_affine_scan",
    "two_stage_associative_scan",
    "SGMobiusKVCache",
    "apply_phase",
    "attention_scores",
    "causal_attention_weights",
    "reversal_phase",
    "rope_tables",
    "nonnegative_feature_map",
    "apply_rotary_angle_addition",
    "inject_mobius_attention",
    "set_model_holonomy_phase",
    "clear_model_holonomy_phase",
]
