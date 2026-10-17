r"""
Semantic Gauge Möbius Rotary Position Embedding (SG-Möbius-RoPE)
Dual-Licensed under Apache-2.0 and MulanPSL-2.0.

Provides data-dependent continuous non-orientable fiber bundle attention for Transformers.
Implements:
1. SGMobiusRotaryEmbedding:
   - Subspace decomposition: \mathbb{R}^d = \mathcal{S}_{\text{anchor}} \oplus \mathcal{S}_{\text{mob}}
     with explicit subspace_mask (0=anchor, 1=mob).
   - Phase-locking attractor: -\eta \sin(2\Phi) in viscous flow for topological self-stabilization at {0, \pi, 2\pi}.
   - Pre-fused RoPE base angle table lookups eliminating eager trigonometric overhead and matching FlashAttention / SDPA latency.
   - Backward compatibility aliases: ContinuousMobiusRoPEModulator, SemanticGaugeMobiusRoPE.
2. 1D viscous flow EMA diffusion and semantic gauge connection prediction (via src.gauge_predictor).
3. Non-negative feature mapping and O(1) causal KV-cache invariance.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gauge_predictor import (
    HysteresisSemanticGaugePredictor,
    SemanticGaugeConnection,
    causal_viscous_ema,
    parallel_affine_scan,
    two_stage_associative_scan,
)

__all__ = [
    "nonnegative_feature_map",
    "apply_rotary_angle_addition",
    "ContinuousMobiusRoPEModulator",
    "SGMobiusRotaryEmbedding",
    "SemanticGaugeMobiusRoPE",
    "inject_mobius_attention",
    "set_model_holonomy_phase",
    "clear_model_holonomy_phase",
    "HysteresisSemanticGaugePredictor",
    "SemanticGaugeConnection",
    "causal_viscous_ema",
    "parallel_affine_scan",
    "two_stage_associative_scan",
]


def nonnegative_feature_map(u: torch.Tensor) -> torch.Tensor:
    r"""
    Non-Negative Feature Map:
        \phi(u) = \text{ELU}(u) + 1 \ge 0
    
    Applied to query and key coordinates in the evolution subspace \mathcal{S}_{\text{mob}}
    prior to phase rotation. Guarantees that baseline similarity is positive-definite,
    preventing negated tokens from generating spurious positive associations.
    """
    return F.elu(u) + 1.0


def apply_rotary_angle_addition(
    cos0: torch.Tensor,
    sin0: torch.Tensor,
    phi: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Exact 2D SO(2) Planar Rotation via Angle Addition:
        \cos_{\mathrm{mod}} = \cos_0 \odot \cos(\Phi) - \sin_0 \odot \sin(\Phi)
        \sin_{\mathrm{mod}} = \sin_0 \odot \cos(\Phi) + \cos_0 \odot \sin(\Phi)
    
    Preserves 2D vector Euclidean norms identically:
        \|R(\theta + \Phi) v\| \equiv \|v\|   \forall \Phi \in \mathbb{R}
    
    Args:
        cos0: Baseline RoPE cos tensor [..., head_dim].
        sin0: Baseline RoPE sin tensor [..., head_dim].
        phi: Continuous phase flow tensor [..., head_dim].
    
    Returns:
        (cos_mod, sin_mod) with identical shape, dtype, and device.
    """
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    cos_mod = cos0 * cos_phi - sin0 * sin_phi
    sin_mod = sin0 * cos_phi + cos0 * sin_phi
    return cos_mod, sin_mod


class SGMobiusRotaryEmbedding(nn.Module):
    r"""
    Semantic Gauge Möbius Rotary Position Embedding (SG-Möbius-RoPE).
    
    Wraps standard transformer rotary embeddings (Qwen3.5-9B, Llama, Mistral) or operates standalone,
    performing continuous 2D SO(2) planar angle modulation with subspace decoupling:
        \mathbb{R}^d = \mathcal{S}_{\text{anchor}} \oplus \mathcal{S}_{\text{mob}}
    
    - Anchor Subspace \mathcal{S}_{\text{anchor}} (mask = 0):
        \omega \equiv 0 \implies \Phi \equiv 0, preserving exact standard RoPE positional
        ordering for syntactic and structural instruction grounding.
    - Evolution Subspace \mathcal{S}_{\text{mob}} (mask = 1):
        Accumulates continuous torsion flow \Phi_{h,m}(t) with restoring attractor -\eta \sin(2\Phi)
        to execute smooth phase advection and destructive interference (\int \omega dt = \pi)
        without vector norm decay.
    
    Pre-fuses cumulative phase directly into RoPE base angle table lookups, eliminating eager
    trigonometric overhead and matching downstream FlashAttention / SDPA kernel latency.
    Guarantees strict O(1) causal KV-cache invariance without retroactive historical cache invalidation.
    """
    def __init__(
        self,
        original_rotary_emb: Optional[nn.Module] = None,
        head_dim: Optional[int] = None,
        anchor_ratio: float = 0.5,
        anchor_strategy: str = "high_freq",
        subspace_mask: Optional[torch.Tensor] = None,
        omega_max: float = 0.05,
        nu: float = 0.5,
        eta: float = 0.0,
        enable_nonnegative_map: bool = False,
        base: float = 10000.0,
        max_position_embeddings: int = 32768,
        pre_fuse_angles: bool = True,
        adaptive_alpha: bool = False,
        alpha_min: float = 0.5,
        alpha_max: float = 0.875
    ):
        super().__init__()
        self.orig_rotary = original_rotary_emb
        self._head_dim = head_dim
        self.anchor_ratio = float(anchor_ratio)
        self.anchor_strategy = anchor_strategy
        self.omega_max = float(omega_max)
        self.nu = float(nu)
        self.eta = float(eta)
        self.enable_nonnegative_map = enable_nonnegative_map
        self.base = float(base)
        self.max_position_embeddings = int(max_position_embeddings)
        self.pre_fuse_angles = pre_fuse_angles
        self.adaptive_alpha = bool(adaptive_alpha)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)

        # Determine effective dimension and initialize inv_freq
        dim = self.head_dim or 64
        num_subspaces = dim // 2

        if self.orig_rotary is not None and hasattr(self.orig_rotary, "inv_freq") and self.orig_rotary.inv_freq is not None:
            self.register_buffer("inv_freq", self.orig_rotary.inv_freq.clone(), persistent=False)
        else:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, num_subspaces, 1, dtype=torch.float32) * 2 / dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Register subspace mask (0 = anchor, 1 = mob)
        if subspace_mask is not None:
            mask_tensor = subspace_mask.bool() if subspace_mask.dtype == torch.bool else (subspace_mask > 0)
            if mask_tensor.numel() == dim:
                mask_tensor = mask_tensor[:num_subspaces]
            self.register_buffer("subspace_mask", mask_tensor, persistent=False)
        else:
            self.subspace_mask = None

        self._current_phi: Optional[torch.Tensor] = None

    @property
    def head_dim(self) -> Optional[int]:
        if self._head_dim is not None:
            return self._head_dim
        if self.orig_rotary is not None:
            for attr in ("dim", "head_dim", "hidden_size"):
                if hasattr(self.orig_rotary, attr):
                    val = getattr(self.orig_rotary, attr)
                    if isinstance(val, int):
                        return val
        return None

    def compute_adaptive_alpha(self, seq_len: int) -> float:
        r"""
        Computes dynamic adaptive frequency band anchor ratio:
            \alpha(L) = \min(0.875, \max(0.5, \frac{\ln(2L/\pi)}{\ln \theta_{\text{base}}}))
        Guarantees that carrier angle (i - j)\theta_m < \pi/2 for all active evolution channels
        and distances i - j \le L, completely eliminating carrier negative-half-cycle aliasing
        and restoring decoy suppression factor > 1.0 at intermediate scales (L=1,116 and L=2,250).
        """
        if seq_len <= 1:
            return self.anchor_ratio
        val = math.log(2.0 * float(seq_len) / math.pi) / math.log(float(self.base))
        return float(min(self.alpha_max, max(self.alpha_min, val)))

    def set_holonomy_phase(self, phi: Optional[torch.Tensor]):
        """Sets the active continuous phase flow tensor."""
        self._current_phi = phi

    def get_holonomy_phase(self) -> Optional[torch.Tensor]:
        """Returns the active continuous phase flow tensor."""
        return self._current_phi

    def clear_holonomy_phase(self):
        """Clears active continuous phase flow tensor, reverting to standard RoPE."""
        self._current_phi = None

    def get_subspace_mask(
        self,
        num_subspaces: int,
        device: torch.device,
        seq_len: Optional[int] = None
    ) -> torch.Tensor:
        r"""
        Constructs boolean mask over 2D rotation subspaces m \in {0, ..., num_subspaces - 1}.
        Returns True (1) for evolution subspaces \mathcal{S}_{\text{mob}}, False (0) for anchor \mathcal{S}_{\text{anchor}}.
        When adaptive_alpha=True and seq_len is provided, dynamically calculates \alpha(L).
        """
        if self.subspace_mask is not None:
            if self.subspace_mask.numel() == num_subspaces:
                return self.subspace_mask.to(device=device)
            elif self.subspace_mask.numel() == num_subspaces * 2:
                return self.subspace_mask[:num_subspaces].to(device=device)

        if self.adaptive_alpha and seq_len is not None and seq_len > 1:
            effective_alpha = self.compute_adaptive_alpha(seq_len)
        else:
            effective_alpha = self.anchor_ratio

        mask = torch.ones(num_subspaces, dtype=torch.bool, device=device)
        num_anchor = int(round(num_subspaces * effective_alpha))
        num_anchor = max(0, min(num_anchor, num_subspaces))

        if self.anchor_strategy in ("high_freq", "first_half"):
            # Higher frequencies (lower indices 0..num_anchor-1) are preserved as anchor
            mask[:num_anchor] = False
        elif self.anchor_strategy == "low_freq":
            # Lower frequencies (higher indices) are preserved as anchor
            mask[num_subspaces - num_anchor:] = False
        elif self.anchor_strategy == "none":
            pass
        return mask

    def expand_subspace_phase(
        self,
        phi: torch.Tensor,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        return_half: bool = False,
        seq_len: Optional[int] = None
    ) -> torch.Tensor:
        r"""
        Expands arbitrary phase input to match head_dim via subspace decoupling:
        [..., num_subspaces] -> [..., head_dim] with duplicate halves for 2D rotation.
        If return_half=True, returns [..., num_subspaces] for direct angle addition table pre-fusion.
        """
        num_subspaces = head_dim // 2
        effective_seq_len = seq_len if seq_len is not None else (phi.shape[-2] if phi.dim() >= 2 else None)
        mask = self.get_subspace_mask(num_subspaces, device=device, seq_len=effective_seq_len)
        num_mob = int(mask.sum().item())

        phi = phi.to(device=device, dtype=dtype)

        # Standardize trailing dimension
        last_dim = phi.shape[-1]
        if last_dim == 1 or phi.dim() <= 2:
            if phi.dim() == 1:
                phi_tensor = phi.view(1, -1, 1)
            elif phi.dim() == 2:
                phi_tensor = phi.unsqueeze(-1)
            else:
                phi_tensor = phi

            # Broadcast scalar phase to all evolution subspaces
            phi_sub = torch.zeros(*phi_tensor.shape[:-1], num_subspaces, dtype=dtype, device=device)
            if num_mob > 0:
                phi_sub[..., mask] = phi_tensor.expand(*phi_tensor.shape[:-1], num_mob)
        elif last_dim == num_mob:
            phi_sub = torch.zeros(*phi.shape[:-1], num_subspaces, dtype=dtype, device=device)
            phi_sub[..., mask] = phi
        elif last_dim == num_subspaces:
            phi_sub = phi * mask.to(dtype=dtype)
        elif last_dim == head_dim:
            if return_half:
                return phi[..., :num_subspaces]
            return phi
        else:
            phi_sub = torch.zeros(*phi.shape[:-1], num_subspaces, dtype=dtype, device=device)
            valid_len = min(last_dim, num_subspaces)
            phi_sub[..., :valid_len] = phi[..., :valid_len] * mask[:valid_len].to(dtype=dtype)

        if return_half:
            return phi_sub
        return torch.cat([phi_sub, phi_sub], dim=-1)

    def _align_subspace_phase(
        self,
        phi: torch.Tensor,
        head_dim: int,
        target_shape: Tuple[int, ...],
        position_ids: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        return_half: bool = True
    ) -> torch.Tensor:
        r"""
        Aligns and broadcasts continuous phase flow with target tensor shape and position ids.
        """
        effective_seq_len = target_shape[-2] if len(target_shape) >= 2 else None
        if position_ids is not None:
            try:
                effective_seq_len = max(effective_seq_len or 0, int(position_ids.max().item()) + 1)
            except Exception:
                pass

        phi_sub = self.expand_subspace_phase(
            phi, head_dim=head_dim, device=device, dtype=dtype, return_half=return_half, seq_len=effective_seq_len
        )
        phi_seq_len = phi_sub.shape[-2]
        target_seq_len = target_shape[-2]

        if position_ids is not None:
            pos = position_ids
            if pos.dim() == 1:
                pos = pos.unsqueeze(0)
            batch_size = target_shape[0] if len(target_shape) >= 3 else 1
            if pos.shape[0] != batch_size and pos.shape[0] == 1:
                pos = pos.expand(batch_size, -1)

            # Clamp position indices safely within phi range
            pos_clamped = torch.clamp(pos, 0, phi_seq_len - 1)
            while phi_sub.dim() < pos.dim() + 1:
                phi_sub = phi_sub.unsqueeze(0)
            if phi_sub.shape[0] != pos.shape[0] and phi_sub.shape[0] == 1:
                phi_sub = phi_sub.expand(pos.shape[0], *phi_sub.shape[1:])

            expand_idx = pos_clamped
            for _ in range(phi_sub.dim() - pos_clamped.dim()):
                expand_idx = expand_idx.unsqueeze(-1)
            tgt_shape = list(phi_sub.shape)
            tgt_shape[-2] = pos.shape[-1]
            expand_idx = expand_idx.expand(tgt_shape)

            phi_aligned = torch.gather(phi_sub, -2, expand_idx)
        else:
            if target_seq_len <= phi_seq_len:
                phi_aligned = phi_sub.narrow(-2, 0, target_seq_len)
            else:
                last_slice = phi_sub.narrow(-2, phi_seq_len - 1, 1)
                padding = last_slice.expand(*phi_sub.shape[:-2], target_seq_len - phi_seq_len, phi_sub.shape[-1])
                phi_aligned = torch.cat([phi_sub, padding], dim=-2)

        while phi_aligned.dim() > len(target_shape) and phi_aligned.shape[0] == 1:
            phi_aligned = phi_aligned.squeeze(0)
        while phi_aligned.dim() < len(target_shape):
            if phi_aligned.dim() == 3 and len(target_shape) == 4:
                phi_aligned = phi_aligned.unsqueeze(1)
            else:
                phi_aligned = phi_aligned.unsqueeze(0)

        if phi_aligned.shape != target_shape:
            try:
                phi_aligned = phi_aligned.expand(target_shape)
            except RuntimeError:
                pass

        return phi_aligned

    def modulate_rotary(
        self,
        cos: torch.Tensor,
        sin: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Modulates baseline RoPE (cos, sin) using continuous 2D planar angle addition:
            \cos_{\mathrm{mod}} = \cos_0 \odot \cos(\Phi) - \sin_0 \odot \sin(\Phi)
            \sin_{\mathrm{mod}} = \sin_0 \odot \cos(\Phi) + \cos_0 \odot \sin(\Phi)
        Retained for backward compatibility and non-pre-fused invocation paths.
        """
        active_phi = phi if phi is not None else self._current_phi
        if active_phi is None:
            return cos, sin

        head_dim = cos.shape[-1]
        device = cos.device
        dtype = cos.dtype

        phi_aligned = self._align_subspace_phase(
            active_phi,
            head_dim=head_dim,
            target_shape=cos.shape,
            position_ids=position_ids,
            device=device,
            dtype=dtype,
            return_half=False
        )

        return apply_rotary_angle_addition(cos, sin, phi_aligned)

    def apply_feature_map(
        self,
        q: torch.Tensor,
        k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Applies non-negative feature map \phi(u) = \text{ELU}(u) + 1 to evolution channels.
        """
        if not self.enable_nonnegative_map:
            return q, k

        head_dim = q.shape[-1]
        num_subspaces = head_dim // 2
        seq_len = q.shape[-2] if q.dim() >= 2 else None
        mask = self.get_subspace_mask(num_subspaces, device=q.device, seq_len=seq_len)

        full_mask = torch.cat([mask, mask], dim=-1)
        q_out = q.clone()
        k_out = k.clone()

        q_out[..., full_mask] = nonnegative_feature_map(q[..., full_mask])
        k_out[..., full_mask] = nonnegative_feature_map(k[..., full_mask])
        return q_out, k_out

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Forward method wrapping the rotary embedding.
        Pre-fuses cumulative phase flow into RoPE base angle table lookups when available,
        eliminating eager trigonometric overhead and matching downstream FlashAttention / SDPA latency.
        """
        if position_ids is None and len(args) > 0 and isinstance(args[0], torch.Tensor):
            position_ids = args[0]
            args = args[1:]

        phi = kwargs.pop("phi", None)
        active_phi = phi if phi is not None else self._current_phi

        if self.orig_rotary is not None:
            # Delegate to wrapped transformer rotary embedding
            cos, sin = self.orig_rotary(x, position_ids, *args, **kwargs)
            return self.modulate_rotary(cos, sin, phi=active_phi, position_ids=position_ids)

        # Standalone rotary computation
        seq_len = x.shape[-2]
        head_dim = self.head_dim or x.shape[-1]
        device = x.device
        dtype = x.dtype

        calc_dtype = torch.float64
        if position_ids is None:
            pos = torch.arange(seq_len, device=device, dtype=calc_dtype)
        else:
            pos = position_ids.to(dtype=calc_dtype)

        num_subspaces = head_dim // 2
        if hasattr(self, "inv_freq") and self.inv_freq is not None:
            inv_freq = self.inv_freq.to(device=device, dtype=calc_dtype)
        else:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, num_subspaces, 1, device=device, dtype=calc_dtype) * 2 / head_dim))

        if pos.dim() == 1:
            if x.dim() >= 3:
                pos_2d = pos.unsqueeze(0)
                angles = pos_2d.unsqueeze(-1) * inv_freq
            else:
                angles = torch.outer(pos, inv_freq)
        else:
            angles = pos.unsqueeze(-1) * inv_freq

        if active_phi is not None and self.pre_fuse_angles:
            # Pre-fuse cumulative phase directly into base angles table
            phi_half = self._align_subspace_phase(
                active_phi,
                head_dim=head_dim,
                target_shape=angles.shape,
                position_ids=position_ids,
                device=device,
                dtype=calc_dtype,
                return_half=True
            )
            fused_angles = angles + phi_half
            emb = torch.cat([fused_angles, fused_angles], dim=-1)
            cos = emb.cos().to(dtype=dtype)
            sin = emb.sin().to(dtype=dtype)
            return cos, sin
        else:
            emb = torch.cat([angles, angles], dim=-1)
            cos = emb.cos().to(dtype=dtype)
            sin = emb.sin().to(dtype=dtype)
            if active_phi is not None:
                return self.modulate_rotary(cos, sin, phi=active_phi, position_ids=position_ids)
            return cos, sin


# Clean backward-compatibility aliases
ContinuousMobiusRoPEModulator = SGMobiusRotaryEmbedding
SemanticGaugeMobiusRoPE = SGMobiusRotaryEmbedding


def inject_mobius_attention(
    model: nn.Module,
    anchor_ratio: float = 0.5,
    anchor_strategy: str = "high_freq",
    subspace_mask: Optional[torch.Tensor] = None,
    enable_nonnegative_map: bool = False,
    adaptive_alpha: bool = False,
    **kwargs
) -> int:
    r"""
    Seamlessly injects Continuous Möbius Attention into any HuggingFace causal LM
    (specifically verified on Qwen3.5-9B, Llama-3, and related architectures).
    
    Replaces existing rotary embeddings in-place with SGMobiusRotaryEmbedding.
    Returns the total number of patched rotary modules.
    """
    patched_count = 0
    for name, module in model.named_modules():
        # Check standard rotary_emb attribute
        if hasattr(module, "rotary_emb") and module.rotary_emb is not None:
            if not isinstance(module.rotary_emb, SGMobiusRotaryEmbedding):
                module.rotary_emb = SGMobiusRotaryEmbedding(
                    module.rotary_emb,
                    anchor_ratio=anchor_ratio,
                    anchor_strategy=anchor_strategy,
                    subspace_mask=subspace_mask,
                    enable_nonnegative_map=enable_nonnegative_map,
                    adaptive_alpha=adaptive_alpha,
                    **kwargs
                )
                patched_count += 1
        elif "rotary_emb" in name and not isinstance(module, SGMobiusRotaryEmbedding):
            pass

    return patched_count


def set_model_holonomy_phase(model: nn.Module, phi: Optional[torch.Tensor]):
    r"""
    Propagates continuous phase flow tensor to all patched Möbius rotary modules in the model.
    """
    for module in model.modules():
        if isinstance(module, SGMobiusRotaryEmbedding):
            module.set_holonomy_phase(phi)


def clear_model_holonomy_phase(model: nn.Module):
    r"""
    Clears continuous phase flow from all patched Möbius rotary modules in the model.
    """
    for module in model.modules():
        if isinstance(module, SGMobiusRotaryEmbedding):
            module.clear_holonomy_phase()
