import numpy as np
import pandas as pd

# 基础物理与经济参数
M_TOTAL = 1e8           # 1亿公吨
G0, MU, RE, RGEO = 9.80665, 398600.44, 6378.0, 42164.0
V_ROT_EQUATOR = 0.4651
ALPHA, L_MAX = 1.0, 6000.0

# 电梯与火箭参数
DV_G, ISP_G, U_PORT, N_PORTS = 2.2, 480.0, 179000.0, 3
DV_BASE, ISP_E, LAMBDA_J = 15.2, 450.0, 2.0
C_DRY, C_PROP, PI_E, ETA_E = 3.1e6, 500.0, 0.03, 0.8

# 10个基地纬度
BASES = {"Alaska (USA)": 57.43528, "Vandenberg (USA)": 34.75133, "Starbase (USA)": 25.99700, "Cape Canaveral (USA)": 28.48889, "Wallops (USA)": 37.84333, "Baikonur (KAZ)": 45.96500, "Kourou (GUF)": 5.16900, "Sriharikota (IND)": 13.72000, "Taiyuan (CHN)": 38.84910, "Mahia (NZL)": -39.26085}

def get_kappa_and_cost(dv, isp, is_elevator=False):
    r = np.exp((dv * 1000) / (isp * G0))
    kappa = r * (1 + ALPHA)
    cost = ALPHA * C_DRY + (r - 1) * (1 + ALPHA) * C_PROP
    if is_elevator:
        cost += kappa * ((MU * (1/RE - 1/RGEO) * 1e6) / (ETA_E * 3.6e6) * PI_E)
    return kappa, cost

# 计算电梯与基地的年运力与单位成本
k_g, c_g = get_kappa_and_cost(DV_G, ISP_G, True)
y_p_g = (U_PORT / k_g) * N_PORTS # 三个港口总年运量

base_stats = []
y_p_e_total = 0
for name, lat in BASES.items():
    k, c = get_kappa_and_cost(DV_BASE - V_ROT_EQUATOR * np.cos(np.radians(lat)), ISP_E)
    y_p = (LAMBDA_J * 365 * L_MAX) / k
    base_stats.append({"Name": name, "Cost": c, "Yearly": y_p})
    y_p_e_total += y_p

# 计算 T_min 状态
t_min = M_TOTAL / (y_p_g + y_p_e_total)
results = []
for i in range(1, 4):
    amt = (U_PORT / k_g) * t_min
    results.append({"Facility": f"Elevator {i}", "Amount(t)": amt, "Cost(USD)": amt * c_g})
for b in base_stats:
    amt = b["Yearly"] * t_min
    results.append({"Facility": f"Base: {b['Name']}", "Amount(t)": amt, "Cost(USD)": amt * b["Cost"]})

df = pd.DataFrame(results)
df.to_csv("min_time_scenario_details.csv", index=False)
print(f"Shortest Time: {t_min:.2f} Years, Total Cost: {df['Cost(USD)'].sum()/1e12:.2f} Trillion USD")