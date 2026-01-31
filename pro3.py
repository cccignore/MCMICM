import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. 基础物理与经济参数 (基于 2.md)
# ==========================================
M_TOTAL = 1e8           # 总运输量: 1亿公吨
G0 = 9.80665            # 标准重力加速度
MU = 398600.44          # 地球引力参数
RE = 6378.0             # 地球半径
RGEO = 42164.0          # GEO半径
V_ROT_EQUATOR = 0.4651  # 赤道自转线速度 (km/s)

# 火箭性能与结构参数
ALPHA = 1.0             # 结构系数 (干重/载荷) [新设定]
L_MAX = 6000.0          # 单次发射最大起飞质量 (吨)

# 方法一: 电梯 + GEO二段火箭参数
DV_G = 2.2              # km/s
ISP_G = 480.0           # s
U_PORT = 179000.0       # 单港口年运力上限 (吨)
N_PORTS = 3             # 银河港口数量
ETA_E = 0.8             # 电梯效率
PI_E = 0.03             # 电价 (USD/kWh)

# 方法二: 地面直发火箭参数
DV_BASE = 15.2          # 地面直发基准速度 (km/s)
ISP_E = 450.0           # s
LAMBDA_J = 2.0          # 单基地日发射频率 (次/日)
C_DRY = 3.1e6           # 干质量单位成本 (USD/t)
C_PROP = 500.0          # 燃料单位成本 (USD/t)

# 10个发射基地的纬度数据
BASES = {
    "Alaska (USA)": 57.43528,
    "Vandenberg (USA)": 34.75133,
    "Starbase (USA)": 25.99700,
    "Cape Canaveral (USA)": 28.48889,
    "Wallops (USA)": 37.84333,
    "Baikonur (KAZ)": 45.96500,
    "Kourou (GUF)": 5.16900,
    "Sriharikota (IND)": 13.72000,
    "Taiyuan (CHN)": 38.84910,
    "Mahia (NZL)": -39.26085
}

# ==========================================
# 2. 核心计算函数
# ==========================================
def get_kappa_and_cost(dv, isp, is_elevator=False):
    """根据速度增量和比冲计算质量放大系数和单位成本"""
    r = np.exp((dv * 1000) / (isp * G0))
    kappa = r * (1 + ALPHA)
    # 每吨建材成本 = 干重(ALPHA)*单价 + 燃料((R-1)*(1+ALPHA))*单价
    cost_rocket = ALPHA * C_DRY + (r - 1) * (1 + ALPHA) * C_PROP
    if is_elevator:
        # 加上电费 (提升总质量 kappa 需要的电能)
        delta_epsilon = MU * (1/RE - 1/RGEO) * 1e6
        kwh_per_ton = (1000 * delta_epsilon) / (ETA_E * 3.6e6)
        cost_rocket += kappa * (kwh_per_ton * PI_E)
    return kappa, cost_rocket, r

# ==========================================
# 3. 计算并导出数据
# ==========================================

# (1) 太空电梯细节
kappa_g, cost_g, r_g = get_kappa_and_cost(DV_G, ISP_G, is_elevator=True)
payload_g_yearly = (N_PORTS * U_PORT) / kappa_g
df_elevator = pd.DataFrame({
    "Parameter": ["Kappa_G", "Unit Cost (USD/t)", "Yearly Net Capacity (t)", "Mass Ratio R"],
    "Value": [kappa_g, cost_g, payload_g_yearly, r_g]
})
df_elevator.to_csv("elevator_details.csv", index=False)

# (2) 各基地细节
base_list = []
for name, lat in BASES.items():
    v_rot = V_ROT_EQUATOR * np.cos(np.radians(lat))
    dv_eff = DV_BASE - v_rot
    kappa, cost, r = get_kappa_and_cost(dv_eff, ISP_E)
    max_payload_yearly = (LAMBDA_J * 365 * L_MAX) / kappa
    base_list.append({
        "Base Name": name,
        "Latitude": lat,
        "Eff_DeltaV": dv_eff,
        "Kappa": kappa,
        "Cost_per_t": cost,
        "Max_Yearly_Payload_t": max_payload_yearly
    })
df_bases = pd.DataFrame(base_list).sort_values("Cost_per_t")
df_bases.to_csv("base_details.csv", index=False)

# (3) Pareto 最优前沿计算
total_max_yearly_all = payload_g_yearly + df_bases["Max_Yearly_Payload_t"].sum()
t_min = M_TOTAL / total_max_yearly_all
t_max = M_TOTAL / payload_g_yearly
t_axis = np.linspace(t_min, t_max, 100)

pareto_list = []
for T in t_axis:
    remaining = M_TOTAL
    cost = 0
    # 优先电梯
    m_g = min(remaining, payload_g_yearly * T)
    cost += m_g * cost_g
    remaining -= m_g
    # 依次调用基地
    for _, b in df_bases.iterrows():
        if remaining <= 0: break
        m_b = min(remaining, b['Max_Yearly_Payload_t'] * T)
        cost += m_b * b['Cost_per_t']
        remaining -= m_b
    pareto_list.append({"Time_Years": T, "Total_Cost_TrillionUSD": cost/1e12})

df_pareto = pd.DataFrame(pareto_list)
df_pareto.to_csv("pareto_results.csv", index=False)

# ==========================================
# 4. 绘图展示
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(df_pareto["Time_Years"], df_pareto["Total_Cost_TrillionUSD"], 'b-', linewidth=2)
plt.title("Pareto Frontier: Total Cost vs. Completion Time")
plt.xlabel("Time (Years)")
plt.ylabel("Total Cost (Trillion USD)")
plt.grid(True)
plt.savefig("pareto_plot.png")
print("计算完成！已生成: elevator_details.csv, base_details.csv, pareto_results.csv, pareto_plot.png")