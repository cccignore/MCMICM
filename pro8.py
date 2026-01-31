import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# 1. 基础参数设置
# =============================================================================
M_TOTAL = 1e8             # 总任务量 1亿吨
G0 = 9.80665
MU = 398600.44
RE = 6378.0
RGEO = 42164.0
V_ROT_EQUATOR = 0.4651

# 核心技术参数
ALPHA_G = 0.3             # 轨道端: 结构系数 (极轻量化)
ALPHA_E = 0.6             # 地面端: 结构系数
L_MAX = 6000.0            # 最大起飞质量
DV_G, ISP_G = 2.2, 460.0  # 电梯端: 真空高比冲
DV_BASE, ISP_E = 15.2, 430.0 # 地面端: 综合比冲

# 经济与运营参数
C_DRY, C_PROP = 3.1e6, 500.0
N_PORTS, U_PORT = 3, 179000.0
ETA_E, PI_E = 0.8, 0.03
LAMBDA_J = 2.0            # 地面日发射频率

# 10个基地数据
BASES_DATA = {
    "Alaska (USA)": 57.43528, "Vandenberg (USA)": 34.75133,
    "Starbase (USA)": 25.99700, "Cape Canaveral (USA)": 28.48889,
    "Wallops (USA)": 37.84333, "Baikonur (KAZ)": 45.96500,
    "Kourou (GUF)": 5.16900, "Sriharikota (IND)": 13.72000,
    "Taiyuan (CHN)": 38.84910, "Mahia (NZL)": -39.26085
}

# =============================================================================
# 2. 核心计算函数
# =============================================================================
def get_rocket_stats(dv, isp, alpha, is_elevator=False):
    r = np.exp((dv * 1000) / (isp * G0))
    kappa = r * (1 + alpha)
    cost = alpha * C_DRY + (r - 1) * (1 + alpha) * C_PROP
    if is_elevator:
        delta_epsilon = MU * (1/RE - 1/RGEO) * 1e6
        kwh_per_ton = (1000 * delta_epsilon) / (ETA_E * 3.6e6)
        cost += kappa * (kwh_per_ton * PI_E)
    return kappa, cost

# --- 初始化数据 ---
k_g, cost_g = get_rocket_stats(DV_G, ISP_G, ALPHA_G, True)
yearly_payload_g = (N_PORTS * U_PORT) / k_g

base_list = []
for name, lat in BASES_DATA.items():
    v_rot = V_ROT_EQUATOR * np.cos(np.radians(lat))
    dv_eff = DV_BASE - v_rot
    k, c = get_rocket_stats(dv_eff, ISP_E, ALPHA_E, False)
    max_y = (LAMBDA_J * 365 * L_MAX) / k
    base_list.append({"Cost": c, "Yearly": max_y})
df_bases = pd.DataFrame(base_list).sort_values("Cost")

total_yearly_rockets = df_bases["Yearly"].sum()

# =============================================================================
# 3. 计算帕累托前沿 (Method 3: 混合调度)
# =============================================================================
T_MIN = M_TOTAL / (yearly_payload_g + total_yearly_rockets) # 最快: 电梯+全火箭
T_MAX = M_TOTAL / yearly_payload_g                          # 最慢: 仅电梯
t_axis = np.linspace(T_MIN, T_MAX, 100)

pareto_data = []
for T in t_axis:
    rem, raw_cost = M_TOTAL, 0
    
    # 贪心算法: 先用便宜的电梯
    m_g = min(rem, yearly_payload_g * T)
    raw_cost += m_g * cost_g
    rem -= m_g
    
    # 再用基地 (按成本排序)
    for _, b in df_bases.iterrows():
        if rem <= 0: break
        m_b = min(rem, b["Yearly"] * T)
        raw_cost += m_b * b["Cost"]
        rem -= m_b
    
    # --- 关键: 统一的压力惩罚公式 ---
    # 越接近 T_MIN，压力越大，成本溢价越高
    pressure = (T_MAX - T) / (T_MAX - T_MIN)
    final_cost = raw_cost * (1 + 0.5 * (pressure ** 2)) # 50% 惩罚系数
    
    pareto_data.append({"Time": T, "Cost": final_cost / 1e12})

df_p = pd.DataFrame(pareto_data)

# =============================================================================
# 4. 计算两个独立的锚点 (纯电梯 / 纯火箭)
# =============================================================================

# --- 点A: 纯方法一 (仅电梯) ---
# 位于 T_MAX，压力为 0，无惩罚
t_m1 = T_MAX
cost_m1 = (M_TOTAL * cost_g) / 1e12 

# --- 点B: 纯方法二 (仅火箭) [修正逻辑] ---
t_m2 = M_TOTAL / total_yearly_rockets

# 1. 计算纯火箭的“裸成本” (Raw Cost)
rem, raw_cost_m2 = M_TOTAL, 0
for _, b in df_bases.iterrows():
    m_b = min(rem, b["Yearly"] * t_m2)
    raw_cost_m2 += m_b * b["Cost"]
    rem -= m_b

# 2. 计算纯火箭面临的“工期压力”
# 注意: 这里的参照系依然是全系统的 T_MAX 和 T_MIN
pressure_m2 = (T_MAX - t_m2) / (T_MAX - T_MIN)

# 3. [核心修正] 施加同样的惩罚
# 这样才能和帕累托曲线公平比较
final_cost_m2 = raw_cost_m2 * (1 + 0.5 * (pressure_m2 ** 2))
cost_m2_trillion = final_cost_m2 / 1e12

# =============================================================================
# 5. 绘图
# =============================================================================
plt.figure(figsize=(12, 7))

# 画帕累托曲线 (蓝色)
plt.plot(df_p["Time"], df_p["Cost"], 'b-', linewidth=3, label='Method 3: Hybrid Pareto Frontier')
plt.fill_between(df_p["Time"], df_p["Cost"], color='blue', alpha=0.1)

# 画点A (绿点 - 纯电梯)
plt.scatter(t_m1, cost_m1, color='green', s=120, zorder=5, label='Method 1: Elevator Only')
plt.annotate(f' ', 
             (t_m1, cost_m1), xytext=(-30, 15), textcoords='offset points', color='green', fontweight='bold')

# 画点B (红点 - 纯火箭 - 已修正)
plt.scatter(t_m2, cost_m2_trillion, color='red', s=120, zorder=5, label='Method 2: Rockets Only (With Penalty)')
plt.annotate(f' ', 
             (t_m2, cost_m2_trillion), xytext=(30, 15), textcoords='offset points', color='red', fontweight='bold')

# 图表装饰
plt.title("Fair Comparison: Hybrid Strategy vs. Single Methods (Standardized Penalty)", fontsize=14)
plt.xlabel("Completion Time (Years)", fontsize=12)
plt.ylabel("Total Project Cost (Trillion USD)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', frameon=True, shadow=True)

plt.tight_layout()
plt.savefig("corrected_comparison_chart.png")
plt.show()

# 打印数值验证
print(f"--- 验证数据 ---")
print(f"方法二(纯火箭)工期: {t_m2:.2f} 年")
print(f"方法二(纯火箭)裸成本: {raw_cost_m2/1e12:.2f} 兆美元")
print(f"方法二(纯火箭)惩罚后成本: {cost_m2_trillion:.2f} 兆美元")
print(f"对应同时间的帕累托(混合)成本: {np.interp(t_m2, df_p['Time'][::-1], df_p['Cost'][::-1]):.2f} 兆美元")