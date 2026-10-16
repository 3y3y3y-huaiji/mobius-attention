r"""
Hysteresis-Gated Semantic Gauge Connection Predictor
Dual-Licensed under Apache-2.0 and MulanPSL-2.0.

Provides:
1. causal_viscous_ema: 1D viscous flow diffusion filter with phase-locking attractor -eta*sin(2*Phi)
   and Lipschitz envelope bounding.
2. SemanticGaugeConnection: Data-dependent continuous semantic gauge connection with parallel prefill
   and stateful O(1) autoregressive decoding.
3. HysteresisSemanticGaugePredictor: Refractory hysteresis-gated gauge predictor suppressing
   spurious multi-trigger parity oscillations across dense contrastive contexts.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_viscous_ema(
    g: torch.Tensor,
    nu: float = 0.5,
    initial_omega: Optional[torch.Tensor] = None,
    eta: float = 0.0,
    omega_max: Optional[float] = None,
    initial_phi: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""
    1D Viscous Flow EMA Diffusion Filter with Phase-Locking Attractor.
    
    Filters raw semantic projection g(t) along the causal sequence dimension with optional
    restoring potential attractor force A(t-1) = -\eta \sin(2 \Phi(t-1)):
        \omega(t) = (1 - \nu) \omega(t-1) + \nu g(t) - \eta \sin(2 \Phi(t-1))
        \omega(t) = \mathrm{clamp}(\omega(t), -\omega_{\max}, +\omega_{\max})
        \Phi(t)   = \Phi(t-1) + \omega(t)
    
    Guarantees:
    - Bounded rate of change |\omega(t)| <= \omega_{\max}.
    - Operator Lipschitz continuity ||\tilde{R}(t+1) - \tilde{R}(t)||_2 <= 2\sin((\theta_m + \omega_{\max})/2).
    - Topological phase self-stabilization at {0, \pi, 2\pi} when \eta > 0.
    - Exact equivalence between sequence prefill and token-by-token stateful decode (< 1e-12).
    
    Args:
        g: Raw projection tensor of shape [..., seq_len, dim] or [..., seq_len].
        nu: Viscous smoothing parameter in (0, 1].
        initial_omega: Optional initial torsion rate from previous state [..., dim].
        eta: Attractor coupling coefficient (\eta >= 0). Default 0.0 (unforced viscous flow).
        omega_max: Optional maximum continuous torsion rate envelope for Lipschitz clamping.
        initial_phi: Optional initial cumulative phase from previous state [..., dim].
    
    Returns:
        Filtered continuous torsion rate tensor \omega with identical shape to g.
    """
    if nu >= 1.0 and eta == 0.0 and omega_max is None:
        if initial_omega is not None and g.shape[-2] == 1:
            return g
        return g

    seq_dim = -2 if g.dim() >= 2 else 0
    seq_len = g.shape[seq_dim]

    if seq_len == 1:
        # O(1) single-step autoregressive decode
        curr_g = g.select(seq_dim, 0)
        if initial_omega is not None:
            prev = initial_omega
            if eta > 0.0 and initial_phi is not None:
                attractor = -eta * torch.sin(2.0 * initial_phi)
            else:
                attractor = 0.0
            out = (1.0 - nu) * prev + nu * curr_g + attractor
        else:
            if eta > 0.0 and initial_phi is not None:
                attractor = -eta * torch.sin(2.0 * initial_phi)
            else:
                attractor = 0.0
            out = nu * curr_g + attractor

        if omega_max is not None:
            out = torch.clamp(out, -omega_max, omega_max)
        return out.unsqueeze(seq_dim)

    # Causal sequence prefill: recurrence guarantees exact equivalence with token-by-token decoding
    alpha = 1.0 - nu
    omega = torch.empty_like(g)
    if initial_omega is not None:
        prev_omega = initial_omega
    else:
        prev_omega = torch.zeros_like(g.select(seq_dim, 0))

    if initial_phi is not None:
        prev_phi = initial_phi
    else:
        prev_phi = torch.zeros_like(g.select(seq_dim, 0))

    for t in range(seq_len):
        curr_g = g.select(seq_dim, t)
        if eta > 0.0:
            attractor = -eta * torch.sin(2.0 * prev_phi)
            step_omega = alpha * prev_omega + nu * curr_g + attractor
        else:
            step_omega = alpha * prev_omega + nu * curr_g

        if omega_max is not None:
            step_omega = torch.clamp(step_omega, -omega_max, omega_max)

        prev_omega = step_omega
        prev_phi = prev_phi + step_omega
        omega.select(seq_dim, t).copy_(step_omega)

    return omega


class SemanticGaugeConnection(nn.Module):
    r"""
    Data-Dependent Continuous Semantic Gauge Connection with Phase-Locking Attractor.
    
    Projects hidden representations to smooth local torsion rate \omega_{h,m}(t) in [-\omega_{\max}, \omega_{\max}]
    via 1D viscous flow diffusion coupled with topological restoring potential -\eta \sin(2\Phi),
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
        eta: float = 0.0,
        normalize_input: bool = True,
        polar_direction: Optional[torch.Tensor] = None,
        bias: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.omega_max = float(omega_max)
        self.nu = float(nu)
        self.eta = float(eta)
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
        """Resets internal autoregressive decoding state buffers."""
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

        weight = self.proj.weight.to(dtype=x_norm.dtype)
        bias = self.proj.bias.to(dtype=x_norm.dtype) if self.proj.bias is not None else None
        g_t = self.omega_max * torch.tanh(F.linear(x_norm, weight, bias))

        if self._last_omega is not None:
            prev_omega = self._last_omega
            prev_phi = self._last_phi if self._last_phi is not None else torch.zeros_like(prev_omega)
            attractor = -self.eta * torch.sin(2.0 * prev_phi) if self.eta > 0.0 else 0.0
            omega_t = (1.0 - self.nu) * prev_omega + self.nu * g_t + attractor
            omega_t = torch.clamp(omega_t, -self.omega_max, self.omega_max)
            phi_t = prev_phi + omega_t
        else:
            omega_t = self.nu * g_t
            omega_t = torch.clamp(omega_t, -self.omega_max, self.omega_max)
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
        weight = self.proj.weight.to(dtype=x_norm.dtype)
        bias = self.proj.bias.to(dtype=x_norm.dtype) if self.proj.bias is not None else None
        raw_g = self.omega_max * torch.tanh(F.linear(x_norm, weight, bias))

        # Viscous flow EMA diffusion smoothing with optional phase attractor
        omega = causal_viscous_ema(
            raw_g,
            nu=self.nu,
            initial_omega=initial_omega,
            eta=self.eta,
            omega_max=self.omega_max,
            initial_phi=initial_phi
        )

        # Cumulative phase path-integral summation
        phi = torch.cumsum(omega, dim=-2)
        if initial_phi is not None:
            phi = phi + initial_phi.unsqueeze(-2)

        if update_state:
            self._last_omega = omega.select(-2, -1).detach()
            self._last_phi = phi.select(-2, -1).detach()

        return omega, phi


class HysteresisSemanticGaugePredictor(SemanticGaugeConnection):
    r"""
    Refractory Hysteresis-Gated Semantic Gauge Connection Predictor.
    
    Suppresses spurious multi-trigger parity oscillations across dense contrastive contexts
    by enforcing a refractory gating window and respecting quotation scopes.
    """
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int = 1,
        omega_max: float = 0.05,
        nu: float = 0.5,
        eta: float = 0.0,
        refractory_window: int = 16,
        normalize_input: bool = True,
        polar_direction: Optional[torch.Tensor] = None,
        bias: bool = True
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            omega_max=omega_max,
            nu=nu,
            eta=eta,
            normalize_input=normalize_input,
            polar_direction=polar_direction,
            bias=bias
        )
        self.refractory_window = int(refractory_window)
        self.register_buffer("_last_trigger_step", torch.tensor(-100000, dtype=torch.long), persistent=False)

    def reset_state(self):
        super().reset_state()
        self._last_trigger_step.fill_(-100000)

    def filter_with_hysteresis(self, raw_triggers: List[int], seq_len: int) -> List[int]:
        r"""
        Applies refractory hysteresis gating:
        Once a contrastive trigger fires at token t, subsequent triggers within
        [t, t + refractory_window] are suppressed to prevent burst parity oscillations.
        """
        filtered: List[int] = []
        last_fired = -self.refractory_window - 1
        for idx in sorted(raw_triggers):
            if idx >= seq_len:
                continue
            if idx - last_fired > self.refractory_window:
                filtered.append(idx)
                last_fired = idx
        return filtered

    def filter_with_quotation_scope(self, text_tokens: List[str], raw_triggers: List[int]) -> List[int]:
        r"""
        Filters triggers to respect quotation boundaries:
        Contrastive tokens inside quotation marks ("...", '...', “...”) are treated
        as quoted speech and do not invert global document polarity.
        """
        in_double_quote = False
        in_single_quote = False
        quote_char_double = {'"', '“', '”'}
        quote_char_single = {"'", '‘', '’'}

        scoped_triggers: List[int] = []
        for idx, token in enumerate(text_tokens):
            for ch in token:
                if ch in quote_char_double:
                    in_double_quote = not in_double_quote
                elif ch in quote_char_single:
                    in_single_quote = not in_single_quote

            if idx in raw_triggers:
                if not (in_double_quote or in_single_quote):
                    scoped_triggers.append(idx)
        return scoped_triggers
