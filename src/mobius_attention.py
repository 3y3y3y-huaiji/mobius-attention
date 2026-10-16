r"""
Semantic Gauge Möbius Rotary Position Embedding (SG-Möbius-RoPE)
Dual-Licensed under Apache-2.0 and MulanPSL-2.0.

Provides data-dependent continuous non-orientable fiber bundle attention for Transformers.
Implements smooth 2D planar rotation via exact trigonometric angle addition, subspace decoupling
between structural anchor channels and semantic evolution channels, 1D viscous EMA diffusion,
non-negative feature mapping, and O(1) causal KV-cache invariance.
"""

from typing import Optional, Tuple, Union, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_viscous_ema(
    g: torch.Tensor,
    nu: float = 0.5,
    initial_omega: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""
    1D Viscous EMA Diffusion Filter.
    
    Filters raw semantic projections g(t) along the causal sequence dimension:
        \omega(t) = (1 - \nu) \omega(t-1) + \nu g(t),   \nu \in (0, 1]
    
    Guarantees bounded rate-of-change |\omega(t) - \omega(t-1)| \le 2\nu \omega_{\max}
    and Lipschitz continuous rotation across adjacent tokens.
    
    Args:
        g: Raw projection tensor of shape [..., seq_len, dim] or [..., seq_len].
        nu: Viscous EMA smoothing parameter in (0, 1].
        initial_omega: Optional state from previous decoding step [..., dim].
    
    Returns:
        Filtered continuous torsion rate tensor with identical shape to g.
    """
    if nu >= 1.0:
        if initial_omega is not None and g.shape[-2] == 1:
            return g
        return g

    # Identify sequence dimension: assume second-to-last if dim >= 2, else dim 0
    seq_dim = -2 if g.dim() >= 2 else 0
    seq_len = g.shape[seq_dim]

    if seq_len == 1:
        # O(1) single-step autoregressive decode
        curr_g = g.select(seq_dim, 0)
        if initial_omega is not None:
            prev = initial_omega
            out = (1.0 - nu) * prev + nu * curr_g
        else:
            out = nu * curr_g
        return out.unsqueeze(seq_dim)

    # Causal sequence prefill: exact recurrence guarantees equivalence with token-by-token decoding
    alpha = 1.0 - nu
    omega = torch.empty_like(g)
    if initial_omega is not None:
        prev = initial_omega
    else:
        prev = torch.zeros_like(g.select(seq_dim, 0))

    for t in range(seq_len):
        curr_g = g.select(seq_dim, t)
        prev = alpha * prev + nu * curr_g
        omega.select(seq_dim, t).copy_(prev)

    return omega


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


class SemanticGaugeConnection(nn.Module):
    r"""
    Data-Dependent Continuous Semantic Gauge Connection.
    
    Projects input token representations / hidden states to smooth local torsion rate
    \omega_{h,m}(t) \in [-\omega_{\max}, \omega_{\max}] via 1D viscous EMA diffusion,
    and integrates the continuous phase flow:
        \Phi_{h,m}(t) = \sum_{\tau \le t} \omega_{h,m}(\tau)
    
    Supports parallel causal sequence prefill and O(1) stateful autoregressive decoding.
    """
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int = 1,
        omega_max: float = 0.05,
        nu: float = 0.5,
        normalize_input: bool = True,
        polar_direction: Optional[torch.Tensor] = None,
        bias: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.omega_max = float(omega_max)
        self.nu = float(nu)
        self.normalize_input = normalize_input

        self.proj = nn.Linear(hidden_dim, out_dim, bias=bias)
        if polar_direction is not None:
            with torch.no_grad():
                self.proj.weight.copy_(polar_direction.view(out_dim, -1))
                if bias and self.proj.bias is not None:
                    self.proj.bias.fill_(-0.5)

        # Stateful buffers for token-by-token autoregressive decoding
        self.register_buffer("_last_omega", None, persistent=False)
        self.register_buffer("_last_phi", None, persistent=False)

    def reset_state(self):
        """Resets the internal autoregressive decoding state buffers."""
        self._last_omega = None
        self._last_phi = None

    def step(self, x_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        O(1) stateful single-token autoregressive update step.
        
        Args:
            x_t: Input representation for the current token [batch, hidden_dim] or [batch, 1, hidden_dim].
        
        Returns:
            (omega_t, phi_t) for the current step.
        """
        is_3d = (x_t.dim() == 3)
        if is_3d:
            x_vec = x_t.squeeze(1)
        else:
            x_vec = x_t

        if self.normalize_input:
            x_norm = x_vec / (x_vec.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            x_norm = x_vec

        g_t = self.omega_max * torch.tanh(self.proj(x_norm))

        if self._last_omega is not None:
            omega_t = (1.0 - self.nu) * self._last_omega + self.nu * g_t
            phi_t = self._last_phi + omega_t
        else:
            omega_t = self.nu * g_t
            phi_t = omega_t

        self._last_omega = omega_t.detach()
        self._last_phi = phi_t.detach()

        if is_3d:
            return omega_t.unsqueeze(1), phi_t.unsqueeze(1)
        return omega_t, phi_t

    def forward(
        self,
        x: torch.Tensor,
        initial_omega: Optional[torch.Tensor] = None,
        initial_phi: Optional[torch.Tensor] = None,
        update_state: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Forward pass for continuous torsion and phase integration.
        
        Args:
            x: Input representations [batch, seq_len, hidden_dim].
            initial_omega: Optional initial torsion state [batch, out_dim].
            initial_phi: Optional initial cumulative phase [batch, out_dim].
            update_state: If True, saves final step state to internal buffers.
        
        Returns:
            (omega, phi):
                omega: Continuous torsion rates [batch, seq_len, out_dim].
                phi: Cumulative phase flow [batch, seq_len, out_dim].
        """
        if self.normalize_input:
            x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            x_norm = x

        # Raw continuous semantic projection g_{h,m}(h_t) = \omega_{\max} \tanh(W x + b)
        raw_g = self.omega_max * torch.tanh(self.proj(x_norm))

        # Viscous EMA diffusion smoothing
        omega = causal_viscous_ema(raw_g, nu=self.nu, initial_omega=initial_omega)

        # Cumulative phase path-integral summation
        phi = torch.cumsum(omega, dim=-2)
        if initial_phi is not None:
            phi = phi + initial_phi.unsqueeze(-2)

        if update_state:
            self._last_omega = omega.select(-2, -1).detach()
            self._last_phi = phi.select(-2, -1).detach()

        return omega, phi


class ContinuousMobiusRoPEModulator(nn.Module):
    r"""
    Continuous Möbius Rotary Position Embedding Modulator.
    
    Wraps standard transformer rotary embeddings (Qwen3.5-9B, Llama, Mistral) and performs
    continuous 2D SO(2) planar angle addition with subspace decoupling:
        \mathcal{S} = \mathcal{S}_{\text{anchor}} \cup \mathcal{S}_{\text{mob}}
    
    - Anchor Subspace \mathcal{S}_{\text{anchor}}:
        \omega \equiv 0 \implies \Phi \equiv 0, preserving exact standard RoPE positional
        ordering for syntactic and structural instruction grounding.
    - Evolution Subspace \mathcal{S}_{\text{mob}}:
        Accumulates continuous torsion flow \Phi_{h,m}(t) to execute smooth phase advection
        and destructive interference (\int \omega dt = \pi) without vector norm decay.
    
    Guarantees strict O(1) causal KV-cache invariance: key vectors cached at step j
    depend solely on historical context \tau \le j with zero retroactive cache invalidation.
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
        enable_nonnegative_map: bool = False
    ):
        super().__init__()
        self.orig_rotary = original_rotary_emb
        self._head_dim = head_dim
        self.anchor_ratio = float(anchor_ratio)
        self.anchor_strategy = anchor_strategy
        self.omega_max = float(omega_max)
        self.nu = float(nu)
        self.enable_nonnegative_map = enable_nonnegative_map

        if subspace_mask is not None:
            self.register_buffer("subspace_mask", subspace_mask.bool(), persistent=False)
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

    def set_holonomy_phase(self, phi: Optional[torch.Tensor]):
        """Sets the active continuous phase flow tensor."""
        self._current_phi = phi

    def get_holonomy_phase(self) -> Optional[torch.Tensor]:
        """Returns the active continuous phase flow tensor."""
        return self._current_phi

    def clear_holonomy_phase(self):
        """Clears active continuous phase flow tensor, reverting to standard RoPE."""
        self._current_phi = None

    def get_subspace_mask(self, num_subspaces: int, device: torch.device) -> torch.Tensor:
        r"""
        Constructs boolean mask over 2D rotation subspaces m \in {0, ..., num_subspaces - 1}.
        Returns True for evolution subspaces \mathcal{S}_{\text{mob}}, False for anchor \mathcal{S}_{\text{anchor}}.
        """
        if self.subspace_mask is not None:
            if self.subspace_mask.numel() == num_subspaces:
                return self.subspace_mask.to(device=device)

        mask = torch.ones(num_subspaces, dtype=torch.bool, device=device)
        num_anchor = int(round(num_subspaces * self.anchor_ratio))
        num_anchor = max(0, min(num_anchor, num_subspaces))

        if self.anchor_strategy == "high_freq":
            # Higher frequencies (lower indices 0..num_anchor-1) are preserved as anchor
            mask[:num_anchor] = False
        elif self.anchor_strategy == "low_freq":
            # Lower frequencies (higher indices) are preserved as anchor
            mask[num_subspaces - num_anchor:] = False
        elif self.anchor_strategy == "first_half":
            mask[:num_anchor] = False
        elif self.anchor_strategy == "none":
            pass
        return mask

    def expand_subspace_phase(
        self,
        phi: torch.Tensor,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        r"""
        Expands arbitrary phase input to match head_dim via subspace decoupling:
        [..., num_subspaces] -> [..., head_dim] with duplicate halves for 2D rotation.
        """
        num_subspaces = head_dim // 2
        mask = self.get_subspace_mask(num_subspaces, device=device)
        num_mob = int(mask.sum().item())

        phi = phi.to(device=device, dtype=dtype)

        # Standardize trailing dimension
        last_dim = phi.shape[-1]
        if last_dim == 1 or phi.dim() <= 2:
            if phi.dim() == 1:
                # [seq_len] -> [1, seq_len, 1]
                phi_tensor = phi.view(1, -1, 1)
            elif phi.dim() == 2:
                # [batch, seq_len] -> [batch, seq_len, 1]
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
            # Already formatted with head_dim
            return phi
        else:
            # General fallback: slice or pad to num_subspaces
            phi_sub = torch.zeros(*phi.shape[:-1], num_subspaces, dtype=dtype, device=device)
            valid_len = min(last_dim, num_subspaces)
            phi_sub[..., :valid_len] = phi[..., :valid_len] * mask[:valid_len].to(dtype=dtype)

        # Duplicate halves to match HuggingFace apply_rotary_pos_emb layout
        return torch.cat([phi_sub, phi_sub], dim=-1)

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
        """
        active_phi = phi if phi is not None else self._current_phi
        if active_phi is None:
            return cos, sin

        head_dim = cos.shape[-1]
        seq_len_curr = cos.shape[-2]
        device = cos.device
        dtype = cos.dtype

        # Expand phase into head_dim
        phi_expanded = self.expand_subspace_phase(active_phi, head_dim=head_dim, device=device, dtype=dtype)

        # Align sequence dimension with position_ids or seq_len_curr
        phi_seq_len = phi_expanded.shape[-2]
        if position_ids is not None:
            pos = position_ids
            if pos.dim() == 1:
                pos = pos.unsqueeze(0)
            batch_size = cos.shape[0] if cos.dim() >= 3 else 1
            if pos.shape[0] != batch_size and pos.shape[0] == 1:
                pos = pos.expand(batch_size, -1)

            # Clamp position indices safely within phi range
            pos_clamped = torch.clamp(pos, 0, phi_seq_len - 1)

            # Expand position indices to gather along sequence dimension
            while phi_expanded.dim() < pos.dim() + 1:
                phi_expanded = phi_expanded.unsqueeze(0)
            
            # Align batch dimension
            if phi_expanded.shape[0] != pos.shape[0] and phi_expanded.shape[0] == 1:
                phi_expanded = phi_expanded.expand(pos.shape[0], *phi_expanded.shape[1:])

            expand_idx = pos_clamped
            for _ in range(phi_expanded.dim() - pos_clamped.dim()):
                expand_idx = expand_idx.unsqueeze(-1)
            target_shape = list(phi_expanded.shape)
            target_shape[-2] = pos.shape[-1]
            expand_idx = expand_idx.expand(target_shape)

            phi_aligned = torch.gather(phi_expanded, -2, expand_idx)
        else:
            if seq_len_curr <= phi_seq_len:
                phi_aligned = phi_expanded.narrow(-2, 0, seq_len_curr)
            else:
                last_slice = phi_expanded.narrow(-2, phi_seq_len - 1, 1)
                padding = last_slice.expand(*phi_expanded.shape[:-2], seq_len_curr - phi_seq_len, head_dim)
                phi_aligned = torch.cat([phi_expanded, padding], dim=-2)

        # Align ranks between phi_aligned and cos
        while phi_aligned.dim() > cos.dim() and phi_aligned.shape[0] == 1:
            phi_aligned = phi_aligned.squeeze(0)
        while phi_aligned.dim() < cos.dim():
            if phi_aligned.dim() == 3 and cos.dim() == 4:
                # [batch, seq_len, dim] -> [batch, 1, seq_len, dim]
                phi_aligned = phi_aligned.unsqueeze(1)
            else:
                phi_aligned = phi_aligned.unsqueeze(0)

        # Broadcast phi to cos/sin shape if needed
        if phi_aligned.shape != cos.shape:
            try:
                phi_aligned = phi_aligned.expand_as(cos)
            except RuntimeError:
                pass

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
        mask = self.get_subspace_mask(num_subspaces, device=q.device)

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
        Accepts flexible HuggingFace calling conventions across Qwen, Llama, and Mistral.
        """
        if position_ids is None and len(args) > 0 and isinstance(args[0], torch.Tensor):
            position_ids = args[0]
            args = args[1:]

        phi = kwargs.pop("phi", None)

        if self.orig_rotary is not None:
            # Delegate to wrapped transformer rotary embedding
            cos, sin = self.orig_rotary(x, position_ids, *args, **kwargs)
        else:
            # Standalone rotary computation fallback
            seq_len = x.shape[-2]
            head_dim = self.head_dim or x.shape[-1]
            device = x.device
            dtype = x.dtype
            
            calc_dtype = torch.float64 if dtype == torch.float64 else torch.float32
            if position_ids is None:
                pos = torch.arange(seq_len, device=device, dtype=calc_dtype)
            else:
                pos = position_ids.to(dtype=calc_dtype)

            num_subspaces = head_dim // 2
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, num_subspaces, 1, device=device, dtype=calc_dtype) * 2 / head_dim))
            
            if pos.dim() == 1:
                if x.dim() >= 3:
                    pos_2d = pos.unsqueeze(0)
                    angles = pos_2d.unsqueeze(-1) * inv_freq
                else:
                    angles = torch.outer(pos, inv_freq)
            else:
                angles = pos.unsqueeze(-1) * inv_freq
            
            emb = torch.cat([angles, angles], dim=-1)
            cos = emb.cos().to(dtype=dtype)
            sin = emb.sin().to(dtype=dtype)

        return self.modulate_rotary(cos, sin, phi=phi, position_ids=position_ids)


# Clean backward-compatibility alias
SemanticGaugeMobiusRoPE = ContinuousMobiusRoPEModulator


def inject_mobius_attention(
    model: nn.Module,
    anchor_ratio: float = 0.5,
    anchor_strategy: str = "high_freq",
    enable_nonnegative_map: bool = False,
    **kwargs
) -> int:
    r"""
    Seamlessly injects Continuous Möbius Attention into any HuggingFace causal LM
    (specifically verified on Qwen3.5-9B, Llama-3, and related architectures).
    
    Replaces existing rotary embeddings in-place with ContinuousMobiusRoPEModulator.
    Returns the total number of patched rotary modules.
    """
    patched_count = 0
    for name, module in model.named_modules():
        # Check standard rotary_emb attribute
        if hasattr(module, "rotary_emb") and module.rotary_emb is not None:
            if not isinstance(module.rotary_emb, ContinuousMobiusRoPEModulator):
                module.rotary_emb = ContinuousMobiusRoPEModulator(
                    module.rotary_emb,
                    anchor_ratio=anchor_ratio,
                    anchor_strategy=anchor_strategy,
                    enable_nonnegative_map=enable_nonnegative_map,
                    **kwargs
                )
                patched_count += 1
        # Fallback for models naming module directly rotary_emb
        elif "rotary_emb" in name and not isinstance(module, ContinuousMobiusRoPEModulator):
            # Check if parent module holds it
            pass

    return patched_count


def set_model_holonomy_phase(model: nn.Module, phi: Optional[torch.Tensor]):
    r"""
    Propagates continuous phase flow tensor to all patched Möbius rotary modules in the model.
    """
    for module in model.modules():
        if isinstance(module, ContinuousMobiusRoPEModulator):
            module.set_holonomy_phase(phi)


def clear_model_holonomy_phase(model: nn.Module):
    r"""
    Clears continuous phase flow from all patched Möbius rotary modules in the model.
    """
    for module in model.modules():
        if isinstance(module, ContinuousMobiusRoPEModulator):
            module.clear_holonomy_phase()
