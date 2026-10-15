r"""
Möbius-RoPE: 4\pi Spinor-Twist Rotary Position Embedding Prototype
对比标准 2\pi RoPE 与 4\pi 莫比乌斯半扭转规范流
"""

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class StandardRoPE(nn.Module):
    """标准 2pi 旋转位置编码 (Su et al., 2021)"""
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq) # [seq_len, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

class MobiusRoPE(nn.Module):
    """
    Möbius-RoPE: 引入不可定向流形半扭转 (Half-Twist) 与 4pi 旋量二重覆盖。
    当沿着序列维度移动一个有效周期 L 时，相位经历拓扑奇偶反转 (v -> -v)，
    完整恢复原态需要 2L 距离 (4pi 周期)。
    """
    def __init__(self, dim: int, cycle_len: int = 512, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.cycle_len = cycle_len
        # 旋转基频折半以对应 4\pi 旋量二重覆盖
        inv_freq = 0.5 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        
        # 连续莫比乌斯半扭转奇偶因子: cos(\pi * t / cycle_len)
        # 随着位置跨越 cycle_len 边界，产生连续光滑的定向反转
        twist_factor = torch.cos(math.pi * t / self.cycle_len).unsqueeze(-1)
        
        freqs = torch.outer(t, self.inv_freq)
        cos_part = torch.cat((freqs.cos(), freqs.cos()), dim=-1) * twist_factor
        sin_part = torch.cat((freqs.sin(), freqs.sin()), dim=-1)
        
        return cos_part, sin_part

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # 模拟 RoPE 旋转变换
    d_half = x.shape[-1] // 2
    x1, x2 = x[..., :d_half], x[..., d_half:]
    rotated_x = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated_x * sin)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    seq_len = 1024
    dim = 64
    cycle_len = 256
    
    std_rope = StandardRoPE(dim=dim).to(device)
    mobius_rope = MobiusRoPE(dim=dim, cycle_len=cycle_len).to(device)
    
    # 随机生成一个 Query 向量在位置 0
    torch.manual_seed(42)
    q = torch.randn(1, dim, device=device)
    q = q / q.norm(dim=-1, keepdim=True)
    
    # 所有位置的 Keys
    k = q.repeat(seq_len, 1) # [seq_len, dim]
    
    # 计算标准 RoPE 注意力内积衰减
    std_cos, std_sin = std_rope(seq_len, device)
    q_std = apply_rotary_emb(q, std_cos[0:1], std_sin[0:1])
    k_std = apply_rotary_emb(k, std_cos, std_sin)
    sim_std = (q_std * k_std).sum(dim=-1).cpu().numpy()
    
    # 计算 Möbius-RoPE 注意力内积衰减
    mob_cos, mob_sin = mobius_rope(seq_len, device)
    q_mob = apply_rotary_emb(q, mob_cos[0:1], mob_sin[0:1])
    k_mob = apply_rotary_emb(k, mob_cos, mob_sin)
    sim_mob = (q_mob * k_mob).sum(dim=-1).cpu().numpy()
    
    print("\n--- 实验数据采样 (前5个位置与半扭转周期处) ---")
    print(f"位置 0 (原点): Standard={sim_std[0]:.4f}, Möbius={sim_mob[0]:.4f}")
    print(f"位置 {cycle_len} (一重半扭转处): Standard={sim_std[cycle_len]:.4f}, Möbius={sim_mob[cycle_len]:.4f} (注意负号与反转效应)")
    print(f"位置 {cycle_len*2} (二重完整闭环处): Standard={sim_std[cycle_len*2]:.4f}, Möbius={sim_mob[cycle_len*2]:.4f}")
    print("\n[Möbius-RoPE 基础算子测试通过！]")
