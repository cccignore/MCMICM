import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. 核心参数初始化 (保持与前文一致)
# ==========================================
M_TOTAL = 1e8
G0, MU, RE, RGEO = 9.80665, 398600.44, 6378.0, 42164.0
V_ROT_EQUATOR = 0.4651
ALPHA, L_MAX = 1.0, 6000.0

DV_G, ISP_G, U_PORT, N_PORTS = 2.2, 480.0, 179000.0, 3
DV_BASE, ISP_E, LAMBDA_J = 15.2, 450.0, 2.0
C_DRY, C_PROP, PI_E, ETA_E = 3.1e6, 500.0, 0.03, 0.8

BASES = {"Alaska (USA)": 57.43528, "Vandenberg (USA)": 34.75133, "Starbase (USA)": 25.99700, "Cape Canaveral (USA)": 28.48889, "Wallops (USA)": 37.84333, "Baikonur (KAZ)": 45.96500, "Kourou (GUF)": 5.16900, "Sriharikota (IND)": 13.72000, "Taiyuan (CHN)": 38.84910, "Mahia (NZL)": -39.26085}

def get_kappa_and_cost(dv, isp, is_elevator=False):
    r = np.exp((dv * 1000) / (isp * G0))
    kappa = r * (1 + ALPHA)
    cost = ALPHA * C_DRY + (r - 1) * (1 + ALPHA) * C_PROP
    if is_elevator:
        cost += kappa * ((MU * (1/RE - 1/RGEO) * 1e6) / (ETA_E * 3.6e6) * PI_E)
    return kappa, cost

# 计算性能
k_g, c_g = get_kappa_and_cost(DV_G, ISP_G, True)
p_g_y = (U_PORT / k_g) * N_PORTS
base_stats = []
for name, lat in BASES.items():
    k, c = get_kappa_and_cost(DV_BASE - V_ROT_EQUATOR * np.cos(np.radians(lat)), ISP_E)
    y_p = (LAMBDA_J * 365 * L_MAX) / k
    base_stats.append({"Name": name, "Cost": c, "Yearly": y_p})
df_bases = pd.DataFrame(base_stats).sort_values("Cost")

# ==========================================
# 2. 多样本点分配计算
# ==========================================
T_MIN = M_TOTAL / (p_g_y + df_bases["Yearly"].sum())
T_MAX = M_TOTAL / p_g_y

# 选取四个观察点：100%速度, 80%速度, 50%速度, 30%速度
sample_times = [T_MIN, T_MIN * 1.25, T_MIN * 2.0, T_MIN * 3.0]
all_scenarios = []

for idx, T in enumerate(sample_times):
    rem = M_TOTAL
    scenario_details = []
    
    # 太空电梯分配
    m_g = min(rem, p_g_y * T)
    scenario_details.append({"Scenario": f"Point_{idx+1}", "Time": T, "Facility": "Elevator_Total", "Amount": m_g, "Cost": m_g * c_g})
    rem -= m_g
    
    # 基地分配 (按成本由低到高)
    for _, b in df_bases.iterrows():
        m_b = min(rem, b['Yearly'] * T)
        scenario_details.append({"Scenario": f"Point_{idx+1}", "Time": T, "Facility": f"Base_{b['Name']}", "Amount": m_b, "Cost": m_b * b['Cost']})
        rem -= m_b
    
    all_scenarios.append(pd.DataFrame(scenario_details))

# 合并并导出
df_comparison = pd.concat(all_scenarios)
df_comparison.to_csv("multi_point_allocation_details.csv", index=False)

print(f"最短工期 T_min: {T_MIN:.2f} 年")
print("多点分配明细已导出至: multi_point_allocation_details.csv")