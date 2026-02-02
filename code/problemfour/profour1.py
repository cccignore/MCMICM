import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# =============================================================================
# 1. 基础参数与设置
# =============================================================================
M_TOTAL = 1e8
G0, MU, RE, RGEO = 9.80665, 398600.44, 6378.0, 42164.0
V_ROT_EQUATOR = 0.4651
ALPHA_G, ALPHA_E = 0.3, 0.6
L_MAX = 6000.0
DV_G, ISP_G = 2.2, 460.0
DV_BASE, ISP_E = 15.2, 430.0
C_DRY, C_PROP = 3.1e6, 500.0
N_PORTS, U_PORT = 3, 179000.0
ETA_E, PI_E = 0.8, 0.03
LAMBDA_J = 2.0
SCALE_DISCOUNT = 0.2  # 20% 规模折扣
START_YEAR = 2050

# --- 环境与社会成本参数 ---
# 1. 大气环境 (Atmosphere)
EF_CO2 = 3.0          # kg CO2 / kg fuel
EF_BC = 0.025         # Black Carbon factor (2.5%)
PSI_STRATO = 3.0      # Stratosphere forcing multiplier
GWP_BC = 460.0        # Global Warming Potential of BC
EF_NOX = 0.015        # kg NOx / kg fuel
ODP_NOX = 0.02        # Ozone Depletion Potential of NOx
ODP_BC = 0.0          
SCC_2050 = 300.0      # Social Cost of Carbon ($/ton CO2)
SC_OZONE = 10000.0    # Social Cost of Ozone Depletion ($/ton CFC-eq)

# 2. 资源耗竭 (Resource)
TAX_SCARCITY = 500.0  # Fuel Scarcity Tax ($/ton fuel)
P_ALLOY = 30000.0     # Aerospace Alloy Price ($/ton material)

# 3. 间接成本 (Indirect)
CI_GRID = 0.05        # Grid Carbon Intensity (kg CO2 / kWh)
MAINT_ELE_CO2 = 7500.0 # Elevator Maint Emissions (ton CO2/yr, Total)
MAINT_BASE_CO2 = 5000.0 # Base Maint Emissions (ton CO2/yr per active base)

BASES_DATA = {
    "Alaska (USA)": 57.43528, "Vandenberg (USA)": 34.75133,
    "Starbase (USA)": 25.99700, "Cape Canaveral (USA)": 28.48889,
    "Wallops (USA)": 37.84333, "Baikonur (KAZ)": 45.96500,
    "Kourou (GUF)": 5.16900, "Sriharikota (IND)": 13.72000,
    "Taiyuan (CHN)": 38.84910, "Mahia (NZL)": -39.26085
}

# 绘图配色
colors = {
    'rocket': '#D95F02',   # 砖红
    'elevator': '#1B9E77', # 蓝绿/深绿
    'hybrid': '#7570B3',   # 蓝紫
    'financial': 'gray',   # 基底/财务线颜色
    'social': '#d62728'    # 社会成本线颜色
}

# =============================================================================
# 2. 核心计算函数
# =============================================================================
def get_rocket_stats(dv, isp, alpha, is_elevator=False):
    # 计算物理参数
    r = np.exp((dv * 1000) / (isp * G0))
    kappa = r * (1 + alpha)
    
    # 物理消耗系数 (每吨载荷)
    mu_dry = alpha
    mu_fuel = (r - 1) * (1 + alpha)
    
    cost = alpha * C_DRY + (r - 1) * (1 + alpha) * C_PROP
    
    kwh_per_ton = 0.0
    if is_elevator:
        delta_epsilon = MU * (1/RE - 1/RGEO) * 1e6
        kwh_per_ton = (1000 * delta_epsilon) / (ETA_E * 3.6e6)
        cost += kappa * (kwh_per_ton * PI_E)
        
    return kappa, cost, mu_fuel, mu_dry, kwh_per_ton

# 初始化基础数据
k_g, cost_g, mu_fuel_g, mu_dry_g, kwh_per_ton_g = get_rocket_stats(DV_G, ISP_G, ALPHA_G, True)
yearly_payload_g = (N_PORTS * U_PORT) / k_g

base_list = []
for name, lat in BASES_DATA.items():
    v_rot = V_ROT_EQUATOR * np.cos(np.radians(lat))
    dv_eff = DV_BASE - v_rot
    k, c, mu_f, mu_d, _ = get_rocket_stats(dv_eff, ISP_E, ALPHA_E, False)
    max_y = (LAMBDA_J * 365 * L_MAX) / k
    base_list.append({
        "Cost": c, "Yearly": max_y, "Name": name,
        "Mu_Fuel": mu_f, "Mu_Dry": mu_d
    })
df_bases = pd.DataFrame(base_list).sort_values("Cost")
total_yearly_rockets = df_bases["Yearly"].sum()
# T_MIN 是混合模式的最短时间（电梯+火箭全开）
T_MIN = M_TOTAL / (yearly_payload_g + total_yearly_rockets)
T_MAX = M_TOTAL / yearly_payload_g

# 核心算法: 动态迭代求解器 (积分修正版 + LCA物理统计)
def solve_allocation_with_integral_cost(time_target, total_mass):
    final_marginal_discount = 0.0 
    
    phy_res = {
        "mass_roc": 0.0, "mass_ele": 0.0,
        "fuel_E": 0.0, "dry_E": 0.0,
        "active_bases": 0
    }
    
    for i in range(10):
        avg_discount = final_marginal_discount * 0.5
        effective_base_costs = df_bases["Cost"] * (1 - avg_discount)
        avg_rocket_cost = effective_base_costs.mean()
        
        rem = total_mass
        cost_iter = 0
        mass_ele = 0
        mass_roc = 0
        
        acc_fuel_E = 0.0
        acc_dry_E = 0.0
        active_bases_set = set()
        
        if avg_rocket_cost < cost_g:
            can_take_roc = min(rem, total_yearly_rockets * time_target)
            
            needed_roc = can_take_roc
            for idx, b in df_bases.iterrows():
                if needed_roc <= 0: break
                take = min(needed_roc, b["Yearly"] * time_target)
                cost_iter += take * b["Cost"] * (1 - avg_discount)
                acc_fuel_E += take * b["Mu_Fuel"]
                acc_dry_E += take * b["Mu_Dry"]
                active_bases_set.add(idx)
                needed_roc -= take
                
            mass_roc += can_take_roc
            rem -= can_take_roc
            cost_iter += rem * cost_g
            mass_ele += rem
        else:
            can_take_ele = min(rem, yearly_payload_g * time_target)
            cost_iter += can_take_ele * cost_g
            mass_ele += can_take_ele
            rem -= can_take_ele
            
            needed_roc = rem
            for idx, b in df_bases.iterrows():
                if needed_roc <= 0: break
                take = min(needed_roc, b["Yearly"] * time_target)
                cost_iter += take * b["Cost"] * (1 - avg_discount)
                acc_fuel_E += take * b["Mu_Fuel"]
                acc_dry_E += take * b["Mu_Dry"]
                active_bases_set.add(idx)
                mass_roc += take
                needed_roc -= take
        
        new_marginal_discount = SCALE_DISCOUNT * (mass_roc / total_mass)
        if abs(new_marginal_discount - final_marginal_discount) < 0.0001:
            phy_res["mass_roc"] = mass_roc
            phy_res["mass_ele"] = mass_ele
            phy_res["fuel_E"] = acc_fuel_E
            phy_res["dry_E"] = acc_dry_E
            phy_res["active_bases"] = len(active_bases_set)
            return cost_iter, mass_roc, mass_ele, new_marginal_discount, phy_res
        final_marginal_discount = new_marginal_discount
        
    phy_res["mass_roc"] = mass_roc; phy_res["mass_ele"] = mass_ele
    phy_res["fuel_E"] = acc_fuel_E; phy_res["dry_E"] = acc_dry_E
    phy_res["active_bases"] = len(active_bases_set)
    return cost_iter, mass_roc, mass_ele, final_marginal_discount, phy_res

# 社会成本计算函数
def calculate_social_total(phy_res, duration):
    # 1. 物理量提取
    m_ele = phy_res["mass_ele"]
    fuel_E = phy_res["fuel_E"]
    dry_E = phy_res["dry_E"]
    
    # 电梯二段消耗
    fuel_G = m_ele * mu_fuel_g
    dry_G = m_ele * mu_dry_g
    
    # 电力消耗
    delta_epsilon = MU * (1/RE - 1/RGEO) * 1e6
    kwh_total = (m_ele * k_g * 1000 * delta_epsilon) / (ETA_E * 3.6e6)
    
    # 2. 计算各环境分项
    # (A) 大气
    c_ghg = fuel_E * EF_CO2 * (SCC_2050 / 1000.0) 
    c_bc = fuel_E * EF_BC * GWP_BC * PSI_STRATO * SCC_2050
    c_ozone = fuel_E * ((EF_NOX * ODP_NOX) + (EF_BC * ODP_BC)) * SC_OZONE
    c_atmosphere = c_ghg + c_bc + c_ozone
    
    # (B) 资源
    total_fuel = fuel_E + fuel_G
    total_material = dry_E + dry_G
    c_resource = (total_fuel * TAX_SCARCITY) + (total_material * P_ALLOY)
    
    # (C) 间接
    c_elec = kwh_total * CI_GRID * (SCC_2050 / 1000.0)
    total_maint_co2 = duration * (MAINT_ELE_CO2 + phy_res["active_bases"] * MAINT_BASE_CO2)
    c_maint = total_maint_co2 * SCC_2050
    c_indirect = c_elec + c_maint
    
    return c_atmosphere + c_resource + c_indirect

def calculate_social_breakdown(phy_res, duration):
    # 1. 物理量提取
    m_ele = phy_res["mass_ele"]
    fuel_E = phy_res["fuel_E"]
    dry_E = phy_res["dry_E"]
    
    # 电梯二段消耗
    fuel_G = m_ele * mu_fuel_g
    dry_G = m_ele * mu_dry_g
    
    # 电力消耗
    delta_epsilon = MU * (1/RE - 1/RGEO) * 1e6
    kwh_total = (m_ele * k_g * 1000 * delta_epsilon) / (ETA_E * 3.6e6)
    
    # 2. 计算各环境分项
    # (A) 大气
    c_ghg = fuel_E * EF_CO2 * (SCC_2050 / 1000.0) 
    c_bc = fuel_E * EF_BC * GWP_BC * PSI_STRATO * SCC_2050
    c_ozone = fuel_E * ((EF_NOX * ODP_NOX) + (EF_BC * ODP_BC)) * SC_OZONE
    c_atmosphere = c_ghg + c_bc + c_ozone
    
    # (B) 资源
    total_fuel = fuel_E + fuel_G
    total_material = dry_E + dry_G
    c_scarcity = total_fuel * TAX_SCARCITY
    c_alloy = total_material * P_ALLOY
    c_resource = c_scarcity + c_alloy
    
    # (C) 间接
    c_elec = kwh_total * CI_GRID * (SCC_2050 / 1000.0)
    total_maint_co2 = duration * (MAINT_ELE_CO2 + phy_res["active_bases"] * MAINT_BASE_CO2)
    c_maint = total_maint_co2 * SCC_2050
    c_indirect = c_elec + c_maint
    
    total_external = c_atmosphere + c_resource + c_indirect
    return {
        "Atmosphere": c_atmosphere,
        "Resource": c_resource,
        "Indirect": c_indirect,
        "TotalSocialExtra": total_external
    }

# =============================================================================
# 3. 数据生成
# =============================================================================
print("正在执行计算...")

# --- A. 帕累托前沿计算 ---
t_axis = np.linspace(T_MIN, T_MAX, 200)
pareto_data = []

for T in t_axis:
    raw_fin_cost, m_roc, m_ele, avg_d, phy_res = solve_allocation_with_integral_cost(T, M_TOTAL)
    
    # 计算社会成本
    bd = calculate_social_breakdown(phy_res, T)
    env_cost_raw = bd["TotalSocialExtra"]
    
    # 压力惩罚
    pressure = (T_MAX - T) / (T_MAX - T_MIN)
    penalty = 1 + 0.5 * (pressure ** 2)
    
    fin_raw = raw_fin_cost
    soc_raw = raw_fin_cost + env_cost_raw
    
    fin_final = fin_raw * penalty
    soc_final = soc_raw * penalty
    
    pareto_data.append({
        "Time": T, 
        "Financial": fin_final / 1e12,
        "Social": soc_final / 1e12,
        "Atmosphere": (bd["Atmosphere"] * penalty) / 1e12,
        "Resource": (bd["Resource"] * penalty) / 1e12,
        "Indirect": (bd["Indirect"] * penalty) / 1e12,
        "Res": phy_res 
    })

df_pareto = pd.DataFrame(pareto_data)

# --- B. 关键点计算 ---

# 1. 基底模型 T=150 (Method 3)
t_base = 150.0
raw_fin_base, m_roc_base, m_ele_base, _, _ = solve_allocation_with_integral_cost(t_base, M_TOTAL)
p_base = (T_MAX - t_base) / (T_MAX - T_MIN)
cost_base_fin = (raw_fin_base * (1 + 0.5 * p_base**2)) / 1e12

# 2. Method 1 (Elevator Only)
t_m1 = T_MAX
# (1) 财务点
raw_fin_m1, _, _, _, phy_m1 = solve_allocation_with_integral_cost(t_m1, M_TOTAL)
cost_m1_fin = raw_fin_m1 / 1e12 # T_MAX时penalty=1
# (2) 社会点
env_m1 = calculate_social_total(phy_m1, t_m1)
cost_m1_soc = (raw_fin_m1 + env_m1) / 1e12

# 3. Method 2 (Rocket Only) - 强制计算逻辑修正
# M2 时间是根据全火箭运力计算的，肯定比 T_MIN 长
t_m2 = M_TOTAL / total_yearly_rockets 

# 手动计算纯火箭模式的各项消耗 (强制 m_ele = 0)
# 财务成本
avg_discount_m2 = SCALE_DISCOUNT * 0.5 # 粗略估计平均折扣
c_m2_fin_raw = 0.0
phy_m2 = {
    "mass_roc": M_TOTAL, "mass_ele": 0.0,
    "fuel_E": 0.0, "dry_E": 0.0,
    "active_bases": len(df_bases)
}

# 遍历所有基地累加
for _, b in df_bases.iterrows():
    # 假设所有基地满负荷运行
    base_fraction = b["Yearly"] / total_yearly_rockets
    c_m2_fin_raw += (base_fraction * M_TOTAL) * b["Cost"]
    phy_m2["fuel_E"] += (base_fraction * M_TOTAL) * b["Mu_Fuel"]
    phy_m2["dry_E"] += (base_fraction * M_TOTAL) * b["Mu_Dry"]

c_m2_fin_raw *= (1 - avg_discount_m2)

# 压力惩罚
p_m2 = (T_MAX - t_m2) / (T_MAX - T_MIN)
pen_m2 = 1 + 0.5 * p_m2**2

# (1) M2 财务点
cost_m2_fin = (c_m2_fin_raw * pen_m2) / 1e12

# (2) M2 社会点
env_m2 = calculate_social_total(phy_m2, t_m2)
cost_m2_soc = ((c_m2_fin_raw + env_m2) * pen_m2) / 1e12

# 4. 计算斜率 (Marginal Cost of Time)
d_fin = np.gradient(df_pareto["Financial"], df_pareto["Time"])
d_soc = np.gradient(df_pareto["Social"], df_pareto["Time"])

idx_150 = (np.abs(df_pareto["Time"] - 150.0)).argmin()
slope_base_150 = d_fin[idx_150] 

# 5. 等斜率点
idx_iso_slope = (np.abs(d_soc - slope_base_150)).argmin()
t_iso = df_pareto["Time"].iloc[idx_iso_slope]
cost_iso_soc = df_pareto["Social"].iloc[idx_iso_slope]
phy_res_iso = df_pareto["Res"].iloc[idx_iso_slope]

print(f"基底模型 T=150 斜率: {slope_base_150:.4f} T/Year")
print(f"社会模型等斜率匹配点: T={t_iso:.1f} Year (斜率: {d_soc[idx_iso_slope]:.4f})")

# --- C. 仿真曲线数据生成 ---
def calculate_trajectory_with_scale_discount(duration, target_total_cost, m_ele_total, m_roc_total):
    years = np.linspace(0, duration, 1000)
    costs = []
    
    step_ele = m_ele_total / 1000
    step_roc = m_roc_total / 1000
    
    cumulative_cost = 0.0
    cumulative_roc_mass = 0.0
    
    avg_base_price = 0
    total_cap = df_bases["Yearly"].sum()
    for _, b in df_bases.iterrows():
        avg_base_price += b["Cost"] * (b["Yearly"]/total_cap)
        
    for _ in years:
        costs.append(cumulative_cost / 1e12)
        
        cost_step_ele = step_ele * cost_g
        
        if M_TOTAL > 0: progress = cumulative_roc_mass / M_TOTAL
        else: progress = 0
        current_discount = SCALE_DISCOUNT * progress
        
        current_rocket_price = avg_base_price * (1 - current_discount)
        cost_step_roc = step_roc * current_rocket_price
        
        cumulative_cost += (cost_step_ele + cost_step_roc)
        cumulative_roc_mass += step_roc
        
    if cumulative_cost > 0:
        scale_fix = target_total_cost / (cumulative_cost / 1e12) 
    else:
        scale_fix = 1.0
        
    final_costs = [c * scale_fix for c in costs]
    return years + START_YEAR, final_costs

# 生成两条对比曲线
years_base, costs_base = calculate_trajectory_with_scale_discount(
    t_base, cost_base_fin, m_ele_base, m_roc_base
)

years_iso, costs_iso = calculate_trajectory_with_scale_discount(
    t_iso, cost_iso_soc, phy_res_iso["mass_ele"], phy_res_iso["mass_roc"]
)

# =============================================================================
# 3.5 导出绘图数据 (Excel)
# =============================================================================
df_keypoints = pd.DataFrame([
    {"Method": "Baseline Financial (T=150)", "Time": t_base, "Financial": cost_base_fin, "Social": np.nan},
    {"Method": "Method 1: Elevator Only", "Time": t_m1, "Financial": cost_m1_fin, "Social": cost_m1_soc},
    {"Method": "Method 2: Rockets Only", "Time": t_m2, "Financial": cost_m2_fin, "Social": cost_m2_soc},
    {"Method": "Iso-Slope Equivalent (Social)", "Time": t_iso, "Financial": np.nan, "Social": cost_iso_soc},
])

df_traj_base = pd.DataFrame({"Year": years_base, "AccumulatedCost_T": costs_base})
df_traj_iso = pd.DataFrame({"Year": years_iso, "AccumulatedCost_T": costs_iso})

export_path = "plot_data_export.xlsx"
with pd.ExcelWriter(export_path, engine="openpyxl") as writer:
    df_bases.to_excel(writer, sheet_name="Bases", index=False)
    df_pareto.drop(columns=["Res"]).to_excel(writer, sheet_name="Pareto", index=False)
    df_keypoints.to_excel(writer, sheet_name="KeyPoints", index=False)
    df_traj_base.to_excel(writer, sheet_name="Trajectory_Baseline", index=False)
    df_traj_iso.to_excel(writer, sheet_name="Trajectory_IsoSlope", index=False)

print(f"--- 绘图数据已导出: {export_path} ---")

# =============================================================================
# 4. 绘图 1: 帕累托前沿 (Iso-Slope Analysis + M1/M2 Corrected)
# =============================================================================
plt.figure(figsize=(12, 7))

# 1. 绘制主曲线
plt.plot(df_pareto["Time"], df_pareto["Social"], color=colors['hybrid'], linewidth=3, label='Social Cost Pareto Frontier')
plt.plot(df_pareto["Time"], df_pareto["Financial"], color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Financial Cost Reference')
plt.fill_between(df_pareto["Time"], df_pareto["Financial"], df_pareto["Social"], color='red', alpha=0.1, label='Environmental Externalities Gap')

# 2. 标记基底点 (T=150, Financial)
plt.scatter(t_base, cost_base_fin, color='gray', s=100, zorder=5)
plt.annotate(f'Base Target\n({t_base:.0f}y, ${cost_base_fin:.1f}T)', (t_base, cost_base_fin), 
             xytext=(-20, -40), textcoords='offset points', color='gray', fontweight='bold',
             arrowprops=dict(arrowstyle="->", color='gray'))

# 3. 标记等斜率点 (T=Iso, Social)
plt.scatter(t_iso, cost_iso_soc, color=colors['rocket'], s=120, zorder=5)
plt.annotate(f'Iso-Slope Equivalent\n({t_iso:.1f}y, ${cost_iso_soc:.1f}T)', (t_iso, cost_iso_soc), 
             xytext=(30, 20), textcoords='offset points', color=colors['rocket'], fontweight='bold',
             arrowprops=dict(arrowstyle="->", color=colors['rocket']))

# 4. 标记 Method 1 (Elevator) - 两个点
plt.scatter(t_m1, cost_m1_fin, color='gray', s=80, zorder=4, marker='o', alpha=0.6)
plt.scatter(t_m1, cost_m1_soc, color=colors['elevator'], s=120, zorder=5, label='Method 1: Elevator Only')
plt.annotate("", xy=(t_m1, cost_m1_soc), xytext=(t_m1, cost_m1_fin),
            arrowprops=dict(arrowstyle="->", color=colors['elevator'], linestyle=":", linewidth=1.5))
plt.text(t_m1 - 5, (cost_m1_soc + cost_m1_fin)/2, "+Env Tax", color=colors['elevator'], fontsize=8, ha='right')

# 5. 标记 Method 2 (Rocket) - 两个点 (修正后: 应在上方)
plt.scatter(t_m2, cost_m2_fin, color='gray', s=80, zorder=4, marker='o', alpha=0.6)
plt.scatter(t_m2, cost_m2_soc, color=colors['rocket'], s=120, zorder=5, label='Method 2: Rockets Only')
plt.annotate("", xy=(t_m2, cost_m2_soc), xytext=(t_m2, cost_m2_fin),
            arrowprops=dict(arrowstyle="->", color=colors['rocket'], linestyle=":", linewidth=1.5))
plt.text(t_m2 + 3, (cost_m2_soc + cost_m2_fin)/2, "+Huge Env Tax", color=colors['rocket'], fontsize=8, ha='left')

# 6. 绘制斜率切线示意
def plot_tangent(t_pt, c_pt, slope, length=20, col='k'):
    x = np.array([t_pt - length, t_pt + length])
    y = c_pt + slope * (x - t_pt)
    plt.plot(x, y, color=col, linestyle=':', linewidth=1.5)

plot_tangent(t_base, cost_base_fin, slope_base_150, col='gray')
plot_tangent(t_iso, cost_iso_soc, slope_base_150, col=colors['rocket'])

plt.title("Pareto Frontier: Iso-Slope Analysis (Marginal Cost Equivalence)", fontsize=14)
plt.xlabel("Completion Time (Years)", fontsize=12)
plt.ylabel("Total Cost (Trillion USD)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig("pareto_iso_slope.png")
plt.show()

# =============================================================================
# 4.5 绘图 1.5: 全生命周期成本构成演化图 (LCSC Composition Evolution)
# =============================================================================
plt.figure(figsize=(12, 7))

T = df_pareto["Time"].values

y_fin = df_pareto["Financial"].values
y_atm = df_pareto["Atmosphere"].values
y_res = df_pareto["Resource"].values
y_ind = df_pareto["Indirect"].values

plt.stackplot(
    T,
    y_fin, y_atm, y_res, y_ind,
    labels=["Financial (Direct)", "Atmosphere Damage", "Resource Depletion", "Indirect Impacts"],
    alpha=0.9
)

plt.plot(T, df_pareto["Social"].values, linewidth=2.0, linestyle="--", label="Total Social Cost (check)")

plt.title("LCSC Composition Evolution (Stacked Area): Atmosphere / Resource / Indirect", fontsize=14)
plt.xlabel("Completion Time (Years)", fontsize=12)
plt.ylabel("Cost (Trillion USD)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig("lcsc_composition_evolution.png")
plt.show()

# =============================================================================
# 5. 绘图 2: 成本累积仿真 (对比图)
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

# 曲线 1: 基底模型 @ 150 (财务成本)
ax.plot(years_base, costs_base, color='gray', linestyle='--', linewidth=2.0, label=f'Baseline Financial (T={t_base:.0f}y)')
ax.scatter(years_base[-1], costs_base[-1], color='gray', s=60)

# 曲线 2: 社会模型 @ Iso-Slope (社会成本)
ax.plot(years_iso, costs_iso, color=colors['rocket'], linestyle='-', linewidth=3.0, label=f'Social Cost Equivalent (T={t_iso:.1f}y)')
ax.scatter(years_iso[-1], costs_iso[-1], color=colors['rocket'], s=100, zorder=5, edgecolor='white', linewidth=1.5)

# 标注
ax.annotate(f'${costs_base[-1]:.1f}T', (years_base[-1], costs_base[-1]), 
             xytext=(-30, 10), textcoords="offset points", color='gray', fontweight='bold')
ax.annotate(f'${costs_iso[-1]:.1f}T', (years_iso[-1], costs_iso[-1]), 
             xytext=(-30, 10), textcoords="offset points", color=colors['rocket'], fontweight='bold')

ax.set_title("Cumulative Trajectories: Baseline vs Social Iso-Slope Strategy", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Accumulated Cost (Trillion USD)", fontsize=12)
ax.grid(True, which='major', linestyle='-', alpha=0.3, color='gray')
ax.set_xlim(START_YEAR, START_YEAR + max(t_base, t_iso) * 1.1)
ax.set_ylim(0, max(costs_base[-1], costs_iso[-1]) * 1.1)
ax.legend(loc='upper left', frameon=False, fontsize=11)

# Inset Zoom
axins = inset_axes(ax, width="40%", height="40%", loc=4, borderpad=3) # loc=4 is lower right
axins.plot(years_base, costs_base, color='gray', linestyle='--', linewidth=2.0)
axins.plot(years_iso, costs_iso, color=colors['rocket'], linestyle='-', linewidth=3.0)

# Zoom in on start
axins.set_xlim(2050, 2120)
axins.set_ylim(0, costs_iso[int(len(costs_iso)*(70/t_iso))] * 1.2) # Auto scale Y
axins.grid(True, linestyle=':', alpha=0.4)
axins.set_title("Zoom: Early Phase Comparison", fontsize=9)
mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5", linestyle="--")

plt.tight_layout()
plt.savefig("trajectory_iso_slope.png")
plt.show()

print("--- 等斜率分析绘图完成 ---")