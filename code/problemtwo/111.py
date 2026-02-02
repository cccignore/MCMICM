import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# =============================================================================
# 0. 开关
# =============================================================================
PLOT = True
EXPORT_EXCEL = True
EXCEL_PATH = "plot_data_code1_base.xlsx"

# =============================================================================
# 1. 基础参数与设置（RGEO 以代码五为准）
# =============================================================================
M_TOTAL = 1e8
G0, MU, RE, RGEO = 9.80665, 398600.44, 6378.0, 106378.0  # <-- RGEO updated
V_ROT_EQUATOR = 0.4651
ALPHA_G, ALPHA_E = 0.3, 0.6
L_MAX = 6000.0
DV_G, ISP_G = 2.2, 460.0
DV_BASE, ISP_E = 15.2, 430.0
C_DRY, C_PROP = 3.1e6, 500.0
N_PORTS, U_PORT = 3, 179000.0
ETA_E, PI_E = 0.8, 0.03
LAMBDA_J = 2.0
SCALE_DISCOUNT = 0.2  # 20% 规模折扣（封顶）
START_YEAR = 2050

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
    'hybrid': '#7570B3'    # 蓝紫
}

# =============================================================================
# 2. 通用工具函数（统一压力项口径：max(0, …)）
# =============================================================================
def pressure_factor(t_used, t_min_ref, t_max_ref):
    """
    统一压力项：只惩罚“赶工”（更快更贵），不惩罚“更慢”。
    pressure = max(0, (T_MAX - t_used)/(T_MAX - T_MIN))
    """
    denom = (t_max_ref - t_min_ref)
    if denom <= 0:
        return 1.0
    p = (t_max_ref - t_used) / denom
    p = max(0.0, p)
    return 1.0 + 0.5 * (p ** 2)

def clamp_discount(d):
    return float(min(SCALE_DISCOUNT, max(0.0, d)))

# =============================================================================
# 3. 核心计算函数
# =============================================================================
def get_rocket_stats(dv, isp, alpha, is_elevator=False):
    r = np.exp((dv * 1000) / (isp * G0))
    kappa = r * (1 + alpha)
    cost = alpha * C_DRY + (r - 1) * (1 + alpha) * C_PROP
    if is_elevator:
        delta_epsilon = MU * (1 / RE - 1 / RGEO) * 1e6
        kwh_per_ton = (1000 * delta_epsilon) / (ETA_E * 3.6e6)
        cost += kappa * (kwh_per_ton * PI_E)
    return kappa, cost

# 初始化基础数据
k_g, cost_g = get_rocket_stats(DV_G, ISP_G, ALPHA_G, True)
yearly_payload_g = (N_PORTS * U_PORT) / k_g

base_list = []
for name, lat in BASES_DATA.items():
    v_rot = V_ROT_EQUATOR * np.cos(np.radians(lat))
    dv_eff = DV_BASE - v_rot
    k, c = get_rocket_stats(dv_eff, ISP_E, ALPHA_E, False)
    max_y = (LAMBDA_J * 365 * L_MAX) / k
    base_list.append({"Cost": c, "Yearly": max_y, "Name": name})

df_bases = pd.DataFrame(base_list).sort_values("Cost").reset_index(drop=True)
total_yearly_rockets = df_bases["Yearly"].sum()

T_MIN = M_TOTAL / (yearly_payload_g + total_yearly_rockets)
T_MAX = M_TOTAL / yearly_payload_g

# 核心算法: 动态迭代求解器（规划层，确定性）
def solve_allocation_with_integral_cost(time_target, total_mass):
    final_marginal_discount = 0.0

    for _ in range(10):
        avg_discount = final_marginal_discount * 0.5
        avg_discount = clamp_discount(avg_discount)

        effective_base_costs = df_bases["Cost"] * (1 - avg_discount)
        avg_rocket_cost = float(effective_base_costs.mean())

        rem = total_mass
        cost_iter = 0.0
        mass_ele = 0.0
        mass_roc = 0.0

        if avg_rocket_cost < cost_g:
            # 极端情况优先火箭
            can_take_roc = min(rem, total_yearly_rockets * time_target)
            cost_iter += can_take_roc * avg_rocket_cost
            mass_roc += can_take_roc
            rem -= can_take_roc

            cost_iter += rem * cost_g
            mass_ele += rem
        else:
            # 正常优先电梯
            can_take_ele = min(rem, yearly_payload_g * time_target)
            cost_iter += can_take_ele * cost_g
            mass_ele += can_take_ele
            rem -= can_take_ele

            needed_rocket = rem
            for _, b in df_bases.iterrows():
                if needed_rocket <= 0:
                    break
                can_take = min(needed_rocket, b["Yearly"] * time_target)
                cost_iter += can_take * b["Cost"] * (1 - avg_discount)
                mass_roc += can_take
                needed_rocket -= can_take

        # 学习曲线（规划层无失败：attempted≈delivered=m_roc；基准=全项目M_TOTAL；封顶）
        new_marginal_discount = clamp_discount(SCALE_DISCOUNT * (mass_roc / total_mass))

        if abs(new_marginal_discount - final_marginal_discount) < 1e-4:
            return cost_iter, mass_roc, mass_ele, new_marginal_discount

        final_marginal_discount = new_marginal_discount

    return cost_iter, mass_roc, mass_ele, final_marginal_discount

# =============================================================================
# 4. 生成帕累托与关键点
# =============================================================================
print("Running base scenario (Code 1 aligned)...")

# A) Pareto（基底）
t_axis = np.linspace(T_MIN, T_MAX, 100)
pareto_rows = []
for T in t_axis:
    raw_cost, m_roc, m_ele, _ = solve_allocation_with_integral_cost(T, M_TOTAL)
    final_cost = raw_cost * pressure_factor(T, T_MIN, T_MAX)
    pareto_rows.append({
        "scenario": "base",
        "curve": "base_pareto",
        "x_time_years": T,
        "y_cost_trillion": final_cost / 1e12,
        "x_kind": "planned",
        "seed": np.nan,
        "note": ""
    })
df_pareto_series = pd.DataFrame(pareto_rows)

# B) Key points
# M1: Elevator Only
t_m1 = T_MAX
cost_m1_tr = (M_TOTAL * cost_g) / 1e12

# M2: Rockets Only（规划口径：逐基地上限 + avg_discount 近似 + pressure）
t_m2 = M_TOTAL / total_yearly_rockets
avg_discount_m2 = clamp_discount(SCALE_DISCOUNT * 0.5)  # 原逻辑：取中值近似
c_m2_raw = 0.0
for _, b in df_bases.iterrows():
    c_m2_raw += b["Yearly"] * t_m2 * b["Cost"]
c_m2_raw *= (1 - avg_discount_m2)
final_cost_m2 = c_m2_raw * pressure_factor(t_m2, T_MIN, T_MAX)
cost_m2_tr = final_cost_m2 / 1e12

# M3: Hybrid at T=150
t_m3 = 150.0
c_m3_raw, m_roc_m3, m_ele_m3, _ = solve_allocation_with_integral_cost(t_m3, M_TOTAL)
final_cost_m3 = c_m3_raw * pressure_factor(t_m3, T_MIN, T_MAX)
cost_m3_tr = final_cost_m3 / 1e12

df_pareto_points = pd.DataFrame([
    {"scenario": "base", "method": "M1", "time_years": t_m1, "cost_trillion": cost_m1_tr, "time_kind": "planned", "note": "Elevator only"},
    {"scenario": "base", "method": "M2", "time_years": t_m2, "cost_trillion": cost_m2_tr, "time_kind": "planned", "note": "Rockets only"},
    {"scenario": "base", "method": "M3", "time_years": t_m3, "cost_trillion": cost_m3_tr, "time_kind": "planned", "note": "Hybrid @150y"},
])

# =============================================================================
# 5. 生成累计成本曲线（仍为“形状示意 + 终点校准”，但折扣封顶）
# =============================================================================
def calculate_simulation_data(duration, target_total_cost_usd, m_ele_total, m_roc_total):
    years = np.linspace(0, duration, 1000)

    step_ele = m_ele_total / 1000.0
    step_roc = m_roc_total / 1000.0

    # 平均基地价格（用于画形状）
    total_cap = df_bases["Yearly"].sum()
    avg_base_price = float((df_bases["Cost"] * (df_bases["Yearly"] / total_cap)).sum())

    cumulative_cost = 0.0
    cumulative_roc_mass = 0.0  # 这里无失败：attempted=delivered

    costs_tr = []
    for _ in years:
        costs_tr.append(cumulative_cost / 1e12)

        cost_step_ele = step_ele * cost_g

        progress = (cumulative_roc_mass / M_TOTAL) if M_TOTAL > 0 else 0.0
        discount = clamp_discount(SCALE_DISCOUNT * progress)
        current_rocket_price = avg_base_price * (1 - discount)
        cost_step_roc = step_roc * current_rocket_price

        cumulative_cost += (cost_step_ele + cost_step_roc)
        cumulative_roc_mass += step_roc

    scale_fix = (target_total_cost_usd / cumulative_cost) if cumulative_cost > 0 else 1.0
    final_costs_tr = [c * scale_fix for c in costs_tr]

    return years + START_YEAR, final_costs_tr

# M1 curve
years_m1, traj_m1 = calculate_simulation_data(t_m1, cost_m1_tr * 1e12, M_TOTAL, 0.0)
# M2 curve (终点用 final_cost_m2)
years_m2, traj_m2 = calculate_simulation_data(t_m2, final_cost_m2, 0.0, M_TOTAL)
# M3 curve (终点用 final_cost_m3)
years_m3, traj_m3 = calculate_simulation_data(t_m3, final_cost_m3, m_ele_m3, m_roc_m3)

df_traj_series = pd.concat([
    pd.DataFrame({"scenario": "base", "curve": "M1", "x_year": years_m1, "y_cost_trillion": traj_m1, "seed": np.nan, "note": ""}),
    pd.DataFrame({"scenario": "base", "curve": "M2", "x_year": years_m2, "y_cost_trillion": traj_m2, "seed": np.nan, "note": ""}),
    pd.DataFrame({"scenario": "base", "curve": "M3", "x_year": years_m3, "y_cost_trillion": traj_m3, "seed": np.nan, "note": ""}),
], ignore_index=True)

# 空 band 表（基底无区间）
df_pareto_band = pd.DataFrame(columns=[
    "scenario", "band", "x_time_years", "y_lower_trillion", "y_upper_trillion", "level", "stat_center"
])
df_traj_band = pd.DataFrame(columns=[
    "scenario", "band", "x_year", "y_lower_trillion", "y_upper_trillion", "level"
])

# meta_params
meta = [
    ("scenario", "base_code1_aligned"),
    ("RGEO", RGEO),
    ("SCALE_DISCOUNT", SCALE_DISCOUNT),
    ("START_YEAR", START_YEAR),
    ("M_TOTAL", M_TOTAL),
    ("T_MIN", T_MIN),
    ("T_MAX", T_MAX),
    ("note", "Base deterministic planning; pressure=max(0,..); RGEO aligned to code5; export tidy tables for plotting"),
]
df_meta = pd.DataFrame(meta, columns=["key", "value"])

# =============================================================================
# 6. 导出 Excel（统一画图数据）
# =============================================================================
if EXPORT_EXCEL:
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        df_meta.to_excel(writer, sheet_name="meta_params", index=False)
        df_pareto_series.to_excel(writer, sheet_name="pareto_series", index=False)
        df_pareto_band.to_excel(writer, sheet_name="pareto_band", index=False)
        df_pareto_points.to_excel(writer, sheet_name="pareto_points", index=False)
        df_traj_series.to_excel(writer, sheet_name="traj_series", index=False)
        df_traj_band.to_excel(writer, sheet_name="traj_band", index=False)

    print(f"[OK] Exported plot data to: {EXCEL_PATH}")

# =============================================================================
# 7. 绘图（可选）
# =============================================================================
if PLOT:
    # Plot 1: Pareto
    plt.figure(figsize=(12, 7))
    plt.plot(df_pareto_series["x_time_years"], df_pareto_series["y_cost_trillion"],
             color=colors["hybrid"], linewidth=3, label="Base Pareto (Code1 aligned)")
    plt.fill_between(df_pareto_series["x_time_years"], df_pareto_series["y_cost_trillion"],
                     color=colors["hybrid"], alpha=0.10)

    # Points
    for _, r in df_pareto_points.iterrows():
        c = {"M1": colors["elevator"], "M2": colors["rocket"], "M3": colors["hybrid"]}[r["method"]]
        plt.scatter(r["time_years"], r["cost_trillion"], color=c, s=120, zorder=5)
        plt.annotate(f'{r["method"]}\n({r["time_years"]:.1f}y, ${r["cost_trillion"]:.1f}T)',
                     (r["time_years"], r["cost_trillion"]),
                     xytext=(10, 10), textcoords="offset points",
                     color=c, fontweight="bold", fontsize=9)

    plt.title("Base Pareto Frontier (Aligned)", fontsize=14)
    plt.xlabel("Completion Time (Years)", fontsize=12)
    plt.ylabel("Total Project Cost (Trillion USD)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    # Plot 2: Trajectory
    fig, ax = plt.subplots(figsize=(12, 7))
    # M1
    ax.plot(years_m1, traj_m1, color=colors["elevator"], linewidth=2.5, label="M1: Elevator Only")
    ax.scatter(years_m1[-1], traj_m1[-1], color=colors["elevator"], s=80, zorder=5, edgecolor="white", linewidth=1.5)

    # M2
    ax.plot(years_m2, traj_m2, color=colors["rocket"], linewidth=2.5, label="M2: Rockets Only")
    ax.scatter(years_m2[-1], traj_m2[-1], color=colors["rocket"], s=80, zorder=5, edgecolor="white", linewidth=1.5)

    # M3
    ax.plot(years_m3, traj_m3, color=colors["hybrid"], linewidth=2.5, label="M3: Hybrid Strategy")
    ax.scatter(years_m3[-1], traj_m3[-1], color=colors["hybrid"], s=80, zorder=5, edgecolor="white", linewidth=1.5)

    ax.set_title("Cumulative Cost Trajectories (Aligned)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Accumulated Cost (Trillion USD)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(START_YEAR, START_YEAR + 450)
    ax.legend(loc="upper left", frameon=False)

    # Inset zoom
    axins = inset_axes(ax, width="40%", height="40%", loc=1, borderpad=3)
    axins.plot(years_m1, traj_m1, color=colors["elevator"], linewidth=2.0)
    axins.plot(years_m2, traj_m2, color=colors["rocket"], linewidth=2.0)
    axins.plot(years_m3, traj_m3, color=colors["hybrid"], linewidth=2.0)
    axins.set_xlim(2050, 2220)
    axins.set_ylim(0, max(traj_m2[-1], traj_m3[-1], traj_m1[-1]) * 0.6)
    axins.grid(True, linestyle=":", alpha=0.4)
    axins.set_title("Zoom: 2050-2220", fontsize=9)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--")

    plt.tight_layout()
    plt.show()

print("--- Code1 aligned run complete ---")