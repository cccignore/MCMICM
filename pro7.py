import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# 1. 基础物理与经济参数 (基于 2.md)
# =============================================================================
M_TOTAL = 1e8           # 总建材运量: 1亿公吨
G0 = 9.80665            # 标准重力加速度
MU = 398600.44          # 地球引力参数
RE = 6378.0             # 地球半径
RGEO = 42164.0          # 地球同步轨道半径
V_ROT_EQUATOR = 0.4651  # 赤道自转线速度 (km/s)

# 结构系数优化设定 (体现轨道与地面环境差异)
ALPHA_G = 0.3           # 方法一: 轨道二段火箭结构系数
ALPHA_E = 0.6           # 方法二: 地面直发火箭结构系数
L_MAX = 6000.0          # 单次发射最大起飞质量上限 (吨)

# 方法一: 电梯 + GEO二段火箭参数
DV_G = 2.2              # GEO -> 月球所需速度增量 (km/s)
ISP_G = 460.0           # 比冲 (s)
U_PORT = 179000.0       # 每个银河港口年吞吐量 (吨)
N_PORTS = 3             # 银河港口数量
ETA_E = 0.8             # 电梯效率
PI_E = 0.03             # 电价 (USD/kWh)

# 方法二: 地面直达火箭参数
DV_BASE = 15.2          # 地面 -> 月球基准速度需求 (km/s)
ISP_E = 430.0           # 比冲 (s)
LAMBDA_J = 2.0          # 单个基地日发射频率上限 (次/日)
C_DRY = 3.1e6           # 火箭干重单位成本 (USD/t)
C_PROP = 500.0          # 推进剂单位成本 (USD/t)

# 10个发射基地的纬度数据
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
    """计算质量比R, 放大系数kappa, 每吨建材成本及单次最大载荷"""
    r = np.exp((dv * 1000) / (isp * G0))
    kappa = r * (1 + alpha)
    payload_per_launch = L_MAX / kappa
    # 单位载荷(1吨建材)成本
    cost_rocket = alpha * C_DRY + (r - 1) * (1 + alpha) * C_PROP
    if is_elevator:
        # 电梯提升电费
        delta_epsilon = MU * (1/RE - 1/RGEO) * 1e6
        kwh_per_ton = (1000 * delta_epsilon) / (ETA_E * 3.6e6)
        cost_rocket += kappa * (kwh_per_ton * PI_E)
    return kappa, cost_rocket, payload_per_launch, r

# =============================================================================
# 3. 数据初始化
# =============================================================================
# 计算太空电梯
k_g, cost_g, payload_g, r_g = get_rocket_stats(DV_G, ISP_G, ALPHA_G, True)
yearly_payload_g = (N_PORTS * U_PORT) / k_g

# 计算各发射基地
base_list = []
for name, lat in BASES_DATA.items():
    v_rot = V_ROT_EQUATOR * np.cos(np.radians(lat))
    dv_eff = DV_BASE - v_rot
    k, c, p, r = get_rocket_stats(dv_eff, ISP_E, ALPHA_E, False)
    max_y = (LAMBDA_J * 365 * L_MAX) / k
    base_list.append({
        "Base Name": name, "Unit_Cost_USD_t": c, 
        "Max_Yearly_Payload_t": max_y, "Payload_per_Launch": p, "Kappa": k
    })
df_bases = pd.DataFrame(base_list).sort_values("Unit_Cost_USD_t")

# =============================================================================
# 4. Pareto 曲线与多场景计算
# =============================================================================
T_MIN = M_TOTAL / (yearly_payload_g + df_bases["Max_Yearly_Payload_t"].sum())
T_MAX = M_TOTAL / yearly_payload_g
t_axis = np.linspace(T_MIN, T_MAX, 100)
pareto_data = []

for T in t_axis:
    rem, raw_cost = M_TOTAL, 0
    # 优先电梯
    m_g = min(rem, yearly_payload_g * T)
    raw_cost += m_g * cost_g
    rem -= m_g
    # 依次基地
    for _, b in df_bases.iterrows():
        if rem <= 0: break
        m_b = min(rem, b["Max_Yearly_Payload_t"] * T)
        raw_cost += m_b * b["Unit_Cost_USD_t"]
        rem -= m_b
    # 15%非线性压力惩罚
    pressure = (T_MAX - T) / (T_MAX - T_MIN)
    final_cost = raw_cost * (1 + 0.5* (pressure ** 2))
    pareto_data.append({"Time_Years": T, "Total_Cost_TrillionUSD": final_cost / 1e12})

# 记录具体采样点的分配详情 (100%速度, 80%速度, 50%速度, 仅电梯)
sample_times = [T_MIN, T_MIN * 1.25, T_MIN * 2.0, T_MAX * 0.99]
allocation_records = []
for i, T in enumerate(sample_times):
    rem = M_TOTAL
    m_g = min(rem, yearly_payload_g * T)
    allocation_records.append({"Point": f"P{i+1}", "Facility": "Space Elevator", "Amount_t": m_g, "Cost_USD": m_g * cost_g})
    rem -= m_g
    for _, b in df_bases.iterrows():
        m_b = min(rem, b["Max_Yearly_Payload_t"] * T)
        allocation_records.append({"Point": f"P{i+1}", "Facility": f"Base_{b['Base Name']}", "Amount_t": m_b, "Cost_USD": m_b * b["Unit_Cost_USD_t"]})
        rem -= m_b

# =============================================================================
# 5. 导出结果
# =============================================================================
pd.DataFrame(pareto_data).to_csv("final_pareto_results.csv", index=False)
pd.DataFrame(allocation_records).to_csv("final_multi_point_allocation.csv", index=False)
df_bases.to_csv("final_base_details.csv", index=False)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot([d["Time_Years"] for d in pareto_data], [d["Total_Cost_TrillionUSD"] for d in pareto_data], 'b-', linewidth=2.5)
plt.title("Pareto Frontier: Project Cost vs. Completion Time", fontsize=14)
plt.xlabel("Completion Time (Years)", fontsize=12)
plt.ylabel("Total Cost (Trillion USD)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("pareto_frontier_final.png")

print(f"最短工期: {T_MIN:.2f} 年")
print(f"数据已导出: final_pareto_results.csv, final_multi_point_allocation.csv, final_base_details.csv")