r"""
Semantic Gauge Möbius Rotary Position Embedding (SG-Möbius-RoPE)
Dual-Licensed under Apache-2.0 and MulanPSL-2.0.

Provides data-dependent non-orientable fiber bundle attention for Transformers.
Enables destructive interference (negative association) and double-negation loop recovery
at arbitrary sequence positions.
"""

import math
import torch
import torch.nn as nn

class SemanticGaugeConnection(nn.Module):
    r"""
    Data-Dependent Semantic Gauge Connection:
    Computes local connection curvature \omega_t \in [0, \pi] from token embeddings / hidden states,
    and cumulative path holonomy \Phi_t = \sum_{\tau \le t} \omega_\tau.
    """
    def __init__(self, hidden_dim: int, polar_direction: torch.Tensor = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(hidden_dim, 1, bias=True)
        if polar_direction is not None:
            with torch.no_grad():
                self.proj.weight.copy_(polar_direction.view(1, -1))
                self.proj.bias.fill_(-0.5)

    def forward(self, x: torch.Tensor):
        # x: [batch, seq_len, hidden_dim]
        # Normalized cosine alignment with contrastive polarity axis
        x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        logits = self.proj(x_norm).squeeze(-1) # [batch, seq_len]
        omega = torch.pi * torch.sigmoid(logits)
        phi = torch.cumsum(omega, dim=-1)
        return omega, phi

class SemanticGaugeMobiusRoPE(nn.Module):
    r"""
    Möbius-RoPE operator wrapping standard rotary embeddings.
    Modulates standard 2\pi RoPE with 4\pi spinor double-covering holonomy.
    """
    def __init__(self, original_rotary_emb: nn.Module):
        super().__init__()
        self.orig_rotary = original_rotary_emb
        self._current_phi = None

    def set_holonomy_phase(self, phi: torch.Tensor):
        self._current_phi = phi

    def clear_holonomy_phase(self):
        self._current_phi = None

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, *args, **kwargs):
        cos, sin = self.orig_rotary(x, position_ids, *args, **kwargs)
        if self._current_phi is None:
            return cos, sin
            
        seq_len_curr = cos.shape[-2]
        twist_1d = torch.cos(self._current_phi[:seq_len_curr]).to(dtype=cos.dtype, device=cos.device)
        view_shape = [1] * cos.dim()
        view_shape[-2] = seq_len_curr
        return cos * twist_1d.view(*view_shape), sin

def inject_mobius_attention(model: nn.Module) -> int:
    r"""
    Non-invasively injects Möbius-RoPE into any HuggingFace causal LM.
    Returns the number of patched rotary modules.
    """
    patched_count = 0
    for name, module in model.named_modules():
        if hasattr(module, "rotary_emb") and module.rotary_emb is not None:
            if not isinstance(module.rotary_emb, SemanticGaugeMobiusRoPE):
                module.rotary_emb = SemanticGaugeMobiusRoPE(module.rotary_emb)
                patched_count += 1
        elif "rotary" in module.__class__.__name__.lower() and not isinstance(module, SemanticGaugeMobiusRoPE):
            pass
    return patched_count
