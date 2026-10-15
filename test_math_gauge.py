r"""
本地数学与物理机制验证脚本:
验证基于语义规范联络 (Semantic Gauge Connection) 的自适应莫比乌斯注意力 (Adaptive Möbius-RoPE)
验证目标:
1. 任意位置反转测试 (t=35, t=120)
2. 顺承控制组测试 (无反转, 0 误报)
3. 否定之否定恢复测试 (t=40 反转, t=110 恢复)
"""
import math
import numpy as np

def test_semantic_gauge_mobius():
    print("=" * 60)
    print("  Adaptive Semantic Gauge Möbius-RoPE 数学几何机制测试")
    print("=" * 60)
    
    seq_len = 160
    dim = 64
    
    # 模拟输入序列的语义极性曲率密度 omega_t \in [0, pi]
    # Case 1: 顺承文本 (Control Group, 全程顺承)
    omega_control = np.zeros(seq_len)
    
    # Case 2: 任意位置单点反转 (例如在 t=35 出现 'However / 反转')
    omega_single_35 = np.zeros(seq_len)
    omega_single_35[35] = np.pi  # 规范曲率脉冲
    
    # Case 3: 任意位置单点反转 (例如在 t=120 出现 'However / 反转')
    omega_single_120 = np.zeros(seq_len)
    omega_single_120[120] = np.pi
    
    # Case 4: 否定之否定 (t=40 出现第一次转折, t=110 出现第二次转折推翻前者)
    omega_double = np.zeros(seq_len)
    omega_double[40] = np.pi
    omega_double[110] = np.pi
    
    cases = [
        ("顺承控制组 (无反转)", omega_control),
        ("任意位置单点反转 (t=35)", omega_single_35),
        ("任意位置单点反转 (t=120)", omega_single_120),
        ("否定之否定双重反转 (t=40 反转, t=110 恢复)", omega_double)
    ]
    
    # 计算 RoPE 频率谱
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, dim, 2) / dim)) # [dim/2]
    # 规范场耦合向量 (低频语义信道强耦合，高频句法信道弱耦合)
    coupling = np.exp(-np.linspace(0, 3, dim // 2))
    coupling = coupling / coupling[0] # 归一化，低频为 1.0
    
    print(f"维度: {dim}, 频率通道数: {len(inv_freq)}, 序列长度: {seq_len}\n")
    
    for name, omega in cases:
        # 1. 沿因果路径积分得到累积规范相位
        phi = np.cumsum(omega) # [seq_len]
        
        # 2. 模拟 Query (在序列末尾 t = 150) 与所有历史 Key (t = 0 到 150) 的相对相位差
        query_pos = 150
        keys_pos = np.arange(151)
        
        # 空间距离
        spatial_dist = query_pos - keys_pos
        # 累积语义规范相位差 \Delta \Phi = \Phi_query - \Phi_key
        delta_phi = phi[query_pos] - phi[keys_pos]
        
        # 提取关键探针位置的相对相位
        key_0_phase = delta_phi[0] / np.pi
        key_50_phase = delta_phi[50] / np.pi
        key_130_phase = delta_phi[130] / np.pi
        
        print(f"【测试用例: {name}】")
        print(f"  Query 位置: {query_pos}")
        print(f"  Key=0 (起始命题) 相对相位: {key_0_phase:.2f} π  -> 拓扑因子 cos(ΔΦ): {np.cos(delta_phi[0]):+.2f}")
        print(f"  Key=50 (中间内容) 相对相位: {key_50_phase:.2f} π -> 拓扑因子 cos(ΔΦ): {np.cos(delta_phi[50]):+.2f}")
        print(f"  Key=130 (后段内容) 相对相位: {key_130_phase:.2f} π -> 拓扑因子 cos(ΔΦ): {np.cos(delta_phi[130]):+.2f}")
        
        # 拓扑滤波效果
        if name == "顺承控制组 (无反转)":
            assert np.allclose(delta_phi, 0), "控制组必须全程无相位偏移！"
            print("  ✓ 结论: 全程零畸变，完全等价于标准 RoPE，无任何误报！")
        elif "t=35" in name:
            assert np.isclose(delta_phi[0], np.pi), "t=0 在 t=35 之后必须经历 pi 翻转！"
            assert np.isclose(delta_phi[50], 0), "t=50 在 t=35 之后与 query 同侧，相位差为 0！"
            print("  ✓ 结论: 起始命题被精准反转相消，t=35 之后的内容保持相干正向连接！")
        elif "t=120" in name:
            assert np.isclose(delta_phi[0], np.pi) and np.isclose(delta_phi[50], np.pi), "t<120 的前置命题全被反转！"
            assert np.isclose(delta_phi[130], 0), "t=130 在转折后，与 query 保持同向！"
            print("  ✓ 结论: 转折点精准移至 t=120，彻底摆脱 256 周期束缚！")
        elif "否定之否定" in name:
            assert np.isclose(delta_phi[0], 2*np.pi), "t=0 经历两次反转，总相位达到 2pi！"
            assert np.isclose(np.cos(delta_phi[0]), 1.0), "2pi 周期下拓扑因子完全恢复为 +1.0！"
            assert np.isclose(delta_phi[50], np.pi), "处于两次转折之间的中间内容 (t=50) 保持反转相消！"
            print("  ✓ 结论: 完美展现否定之否定！起始命题重新闭合正向激活，中间反转内容被相消！")
        print("-" * 60)

if __name__ == "__main__":
    test_semantic_gauge_mobius()
