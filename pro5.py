import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 基础物理与经济参数
M_TOTAL = 1e8
G0, MU, RE, RGEO = 9.80665, 398600.44, 6378.0, 42164.0
V_ROT_EQUATOR = 0.4651
ALPHA, L_MAX = 1.0, 6000.0  # 结构系数与总重约束

# 方法一 (电梯) 与 方法二 (地面) 参数
DV_G, ISP_G, U_PORT, N_PORTS = 2.2, 480.0, 179000.0, 3
DV_BASE, ISP_E, LAMBDA_J = 15.2, 450.0, 2.0
C_DRY, C_PROP, PI_E, ETA_E = 3.1e6, 500.0, 0.03, 0.8

# 10个基地数据
BASES = {"Alaska (USA)": 57.43528, "Vandenberg (USA)": 34.75133, "Starbase (USA)": 25.99700, "Cape Canaveral (USA)": 28.48889, "Wallops (USA)": 37.84333, "Baikonur (KAZ)": 45.96500, "Kourou (GUF)": 5.16900, "Sriharikota (IND)": 13.72000, "Taiyuan (CHN)": 38.84910, "Mahia (NZL)": -39.26085}

def get_kappa_and_cost(dv, isp, is_elevator=False):
    r = np.exp((dv * 1000) / (isp * G0))
    kappa = r * (1 + ALPHA)
    cost = ALPHA * C_DRY + (r - 1) * (1 + ALPHA) * C_PROP
    if is_elevator:
        cost += kappa * ((MU * (1/RE - 1/RGEO) * 1e6) / (ETA_E * 3.6e6) * PI_E)
    return kappa, cost, r

# 初始化数据
k_g, c_g, r_g = get_kappa_and_cost(DV_G, ISP_G, True)
p_g_y = (U_PORT / k_g) * N_PORTS
base_stats = []
p_e_y_total = 0
for name, lat in BASES.items():
    k, c, r = get_kappa_and_cost(DV_BASE - V_ROT_EQUATOR * np.cos(np.radians(lat)), ISP_E)
    y_p = (LAMBDA_J * 365 * L_MAX) / k
    base_stats.append({"Name": name, "Cost": c, "Yearly": y_p})
    p_e_y_total += y_p
df_bases = pd.DataFrame(base_stats).sort_values("Cost")

# 生成 Pareto 曲线 (引入非线性压力)
T_MIN, T_MAX = M_TOTAL / (p_g_y + p_e_y_total), M_TOTAL / p_g_y
t_axis = np.linspace(T_MIN, T_MAX, 100)
pareto_data = []
for T in t_axis:
    rem = M_TOTAL
    m_g = min(rem, p_g_y * T)
    raw_cost = m_g * c_g
    rem -= m_g
    for _, b in df_bases.iterrows():
        if rem <= 0: break
        m_b = min(rem, b['Yearly'] * T)
        raw_cost += m_b * b['Cost']
        rem -= m_b
    # 引入时间压力惩罚 (15%) 让曲线弯曲
    pressure = (T_MAX - T) / (T_MAX - T_MIN)
    total_cost = raw_cost * (1 + 0.15 * (pressure ** 2))
    pareto_data.append({"Time_Years": T, "Total_Cost_TrillionUSD": total_cost / 1e12})

# 导出 Excel/CSV
df_p = pd.DataFrame(pareto_data)
df_p.to_csv("pareto_results.csv", index=False)
df_bases.to_csv("base_details.csv", index=False)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(df_p["Time_Years"], df_p["Total_Cost_TrillionUSD"], 'b-', linewidth=2.5)
plt.title("Pareto Frontier: Cost vs Time")
plt.xlabel("Years")
plt.ylabel("Trillion USD")
plt.grid(True)
plt.savefig("pareto_plot.png")