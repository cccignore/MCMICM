import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# =============================================================================
# 0. 开关
# =============================================================================
PLOT = True
EXPORT_EXCEL = True
EXCEL_PATH = "plot_data_code3_reliability.xlsx"

SCENARIO = "s3_reliability_soft"

# =============================================================================
# 1. 基础参数与设置（RGEO 以代码五为准）
# =============================================================================
M_TOTAL = 1e8
G0, MU, RE, RGEO = 9.80665, 398600.44, 6378.0, 106378.0
V_ROT_EQUATOR = 0.4651
ALPHA_G, ALPHA_E = 0.3, 0.6
L_MAX = 6000.0
DV_G, ISP_G = 2.2, 460.0
DV_BASE, ISP_E = 15.2, 430.0
C_DRY, C_PROP = 3.1e6, 500.0
N_PORTS, U_PORT = 3, 179000.0
ETA_E, PI_E = 0.8, 0.03
LAMBDA_J = 2.0
SCALE_DISCOUNT = 0.2
START_YEAR = 2050

BASES_DATA = {
    "Alaska (USA)": 57.43528, "Vandenberg (USA)": 34.75133,
    "Starbase (USA)": 25.99700, "Cape Canaveral (USA)": 28.48889,
    "Wallops (USA)": 37.84333, "Baikonur (KAZ)": 45.96500,
    "Kourou (GUF)": 5.16900, "Sriharikota (IND)": 13.72000,
    "Taiyuan (CHN)": 38.84910, "Mahia (NZL)": -39.26085
}

colors = {
    'rocket': '#D95F02',
    'elevator': '#1B9E77',
    'hybrid': '#7570B3'
}

# =============================================================================
# 2. 基地异质性配置（MTBF/MTTR）
# =============================================================================
TIER_PARAMS = {
    "Modern":   {"mtbf": 300, "mttr": 2},
    "Standard": {"mtbf": 200, "mttr": 4},
    "Legacy":   {"mtbf": 140, "mttr": 7},
}

BASE_TIERS = {
    "Starbase (USA)": "Modern", "Mahia (NZL)": "Modern", "Alaska (USA)": "Modern",
    "Kourou (GUF)": "Standard", "Taiyuan (CHN)": "Standard", "Sriharikota (IND)": "Standard",
    "Vandenberg (USA)": "Legacy", "Wallops (USA)": "Legacy",
    "Cape Canaveral (USA)": "Legacy", "Baikonur (KAZ)": "Legacy"
}

# =============================================================================
# 3. 通用工具函数（压力项统一 + 折扣封顶）
# =============================================================================
def clamp_discount(d):
    return float(min(SCALE_DISCOUNT, max(0.0, d)))

def pressure_factor(t_used, t_min_ref, t_max_ref):
    denom = (t_max_ref - t_min_ref)
    if denom <= 0:
        return 1.0
    p = (t_max_ref - t_used) / denom
    p = max(0.0, p)
    return 1.0 + 0.5 * (p ** 2)

# =============================================================================
# 4. 核心计算函数与基础数据
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

# 电梯：理想（不故障）
k_g, cost_g = get_rocket_stats(DV_G, ISP_G, ALPHA_G, True)
yearly_payload_g = (N_PORTS * U_PORT) / k_g

# 火箭基地：理想产能 + 故障参数
base_configs_list = []
for name, lat in BASES_DATA.items():
    v_rot = V_ROT_EQUATOR * np.cos(np.radians(lat))
    dv_eff = DV_BASE - v_rot
    k, c = get_rocket_stats(dv_eff, ISP_E, ALPHA_E, False)

    max_y_ideal = (LAMBDA_J * 365 * L_MAX) / k

    tier = BASE_TIERS.get(name, "Standard")
    params = TIER_PARAMS[tier]

    base_configs_list.append({
        "Name": name,
        "Cost": c,
        "Yearly_Ideal": max_y_ideal,
        "MTBF": float(params["mtbf"]),
        "MTTR": float(params["mttr"]),
        "Tier": tier
    })

df_bases = pd.DataFrame(base_configs_list).sort_values("Cost").reset_index(drop=True)
total_yearly_rockets_ideal = float(df_bases["Yearly_Ideal"].sum())

# 统一参考：用“理想产能”来定义计划层 T_MIN/T_MAX（压力项也用这一套）
T_MIN = M_TOTAL / (yearly_payload_g + total_yearly_rockets_ideal)
T_MAX = M_TOTAL / yearly_payload_g

# =============================================================================
# 5. 规划层求解器（乐观：只看理想产能，不预扣可靠性）
# =============================================================================
def solve_allocation_planning(time_target, total_mass):
    final_marginal_discount = 0.0

    for _ in range(10):
        avg_discount = clamp_discount(final_marginal_discount * 0.5)
        avg_rocket_cost = float((df_bases["Cost"] * (1 - avg_discount)).mean())

        rem = total_mass
        cost_iter = 0.0
        mass_ele = 0.0
        mass_roc = 0.0

        if avg_rocket_cost < cost_g:
            can_take_roc = min(rem, total_yearly_rockets_ideal * time_target)
            cost_iter += can_take_roc * avg_rocket_cost
            mass_roc += can_take_roc
            rem -= can_take_roc

            cost_iter += rem * cost_g
            mass_ele += rem
        else:
            can_take_ele = min(rem, yearly_payload_g * time_target)
            cost_iter += can_take_ele * cost_g
            mass_ele += can_take_ele
            rem -= can_take_ele

            needed_rocket = rem
            for _, b in df_bases.iterrows():
                if needed_rocket <= 0:
                    break
                can_take = min(needed_rocket, b["Yearly_Ideal"] * time_target)
                cost_iter += can_take * b["Cost"] * (1 - avg_discount)
                mass_roc += can_take
                needed_rocket -= can_take

        new_marginal_discount = clamp_discount(SCALE_DISCOUNT * (mass_roc / total_mass))
        if abs(new_marginal_discount - final_marginal_discount) < 1e-4:
            return cost_iter, mass_roc, mass_ele, new_marginal_discount

        final_marginal_discount = new_marginal_discount

    return cost_iter, mass_roc, mass_ele, final_marginal_discount

# =============================================================================
# 6. 执行层：火箭离散事件仿真（局部 RNG；不做“可用度补偿”）
# =============================================================================
def simulate_rocket_trajectory(target_mass, seed=101, pace_years=None):
    """
    - 按基地理想日运力 daily_caps 运行
    - 每天每基地：先维修判定，再故障抽样，再发射
    - pace_years 给定时：节流到计划配速（不补偿故障，不加班）
    - 学习曲线折扣：按 delivered_mass 近似（本 scenario 不涉及失败尝试量）
    """
    if target_mass <= 0:
        return np.array([0.0]), np.array([0.0]), np.array([0.0])

    rng = np.random.default_rng(seed)

    num_bases = len(df_bases)
    bases_up = np.ones(num_bases, dtype=int)
    repair_left = np.zeros(num_bases, dtype=int)

    daily_caps = df_bases["Yearly_Ideal"].values / 365.0
    unit_costs = df_bases["Cost"].values
    mtbfs = df_bases["MTBF"].values
    mttrs = df_bases["MTTR"].values

    # 节流：按计划配速（不补偿故障）
    if pace_years is not None:
        daily_needed = target_mass / (pace_years * 365.0)
        daily_max = float(np.sum(daily_caps))
        if daily_max > daily_needed:
            throttle = daily_needed / daily_max
            daily_caps = daily_caps * throttle

    days = [0.0]
    costs = [0.0]
    masses = [0.0]

    delivered_mass = 0.0
    total_cost = 0.0
    day_count = 0

    while delivered_mass < target_mass:
        day_count += 1
        delivered_today = 0.0

        for i in range(num_bases):
            # 维修中
            if bases_up[i] == 0:
                repair_left[i] -= 1
                if repair_left[i] <= 0:
                    bases_up[i] = 1
                else:
                    continue

            # 故障抽样（每天一次）
            if mttrs[i] > 0 and mtbfs[i] > 0:
                if rng.random() < (1.0 / mtbfs[i]):
                    bases_up[i] = 0
                    r_days = int(rng.normal(mttrs[i], mttrs[i] * 0.2))
                    repair_left[i] = max(1, r_days)
                    continue

            # 正常发射
            prod = float(daily_caps[i])
            if delivered_mass + delivered_today + prod > target_mass:
                prod = target_mass - (delivered_mass + delivered_today)
            if prod <= 0:
                continue

            midpoint = delivered_mass + delivered_today + prod / 2.0
            discount = clamp_discount(SCALE_DISCOUNT * (midpoint / M_TOTAL))
            total_cost += prod * unit_costs[i] * (1 - discount)
            delivered_today += prod

        delivered_mass += delivered_today

        if (day_count % 30 == 0) or (delivered_mass >= target_mass):
            days.append(day_count / 365.0)
            costs.append(total_cost / 1e12)
            masses.append(delivered_mass)

        if day_count > 1000 * 365:
            break

    return np.array(days), np.array(costs), np.array(masses)

# =============================================================================
# 7. 数据生成：Pareto（规划线）+ Soft deadline（执行单种子）
# =============================================================================
print("Running scenario 3 (Reliability + soft deadline) aligned...")

t_axis = np.linspace(T_MIN, T_MAX * 1.1, 100)

# A) 规划 Pareto（基底 planning）
pareto_series_rows = []
for T in t_axis:
    raw_cost, _, _, _ = solve_allocation_planning(T, M_TOTAL)
    final_cost = raw_cost * pressure_factor(T, T_MIN, T_MAX)
    pareto_series_rows.append({
        "scenario": SCENARIO,
        "curve": "base_pareto_planning",
        "x_time_years": T,
        "y_cost_trillion": final_cost / 1e12,
        "x_kind": "planned",
        "seed": np.nan,
        "note": "Planning (optimistic, no reliability pre-deduction)"
    })

# B) Soft deadline Pareto（执行单种子，按计划配速，允许延期；且不允许提前交付）
SOFT_SEED = 101
soft_rows = []
for T_plan in t_axis:
    # 规划分配（乐观）
    _, m_roc_plan, m_ele_plan, _ = solve_allocation_planning(T_plan, M_TOTAL)

    # 火箭执行：按计划配速节流，不补班
    roc_days, roc_costs, _ = simulate_rocket_trajectory(m_roc_plan, seed=SOFT_SEED, pace_years=T_plan)
    t_roc = float(roc_days[-1])
    cost_roc_usd = float(roc_costs[-1]) * 1e12

    # 电梯执行：理想线性
    t_ele = (m_ele_plan / yearly_payload_g) if yearly_payload_g > 0 else 0.0
    cost_ele_usd = m_ele_plan * cost_g

    # 真实交付：不允许提前完成（至少等到计划点）
    t_real = max(T_plan, t_ele, t_roc)

    raw_cost_real = cost_ele_usd + cost_roc_usd
    final_cost_real = raw_cost_real * pressure_factor(t_real, T_MIN, T_MAX)

    soft_rows.append({
        "scenario": SCENARIO,
        "curve": "soft_deadline_single_seed",
        "x_time_years": t_real,          # 横轴用真实完工时间（更符合“真实交付”）
        "y_cost_trillion": final_cost_real / 1e12,
        "x_kind": "real",
        "seed": SOFT_SEED,
        "note": "Execution (stochastic rockets) w/ no-early-finish"
    })

df_pareto_series = pd.DataFrame(pareto_series_rows + soft_rows)

# band 表为空（本 scenario 单种子）
df_pareto_band = pd.DataFrame(columns=[
    "scenario", "band", "x_time_years", "y_lower_trillion", "y_upper_trillion", "level", "stat_center"
])

# =============================================================================
# 8. 关键点与累计成本曲线（M1/M2/M3）
# =============================================================================
# M1：电梯 only（理想线性）
t_m1 = T_MAX
cost_m1_tr = (M_TOTAL * cost_g) / 1e12
years_m1 = np.linspace(START_YEAR, START_YEAR + t_m1, 200)
traj_m1 = np.linspace(0.0, cost_m1_tr, 200)

# M2：火箭 only（执行随机故障单种子）
print("  Simulating M2 stochastic...")
m2_days, m2_costs_tr, _ = simulate_rocket_trajectory(M_TOTAL, seed=SOFT_SEED, pace_years=None)
t_m2_real = float(m2_days[-1])
raw_cost_m2 = float(m2_costs_tr[-1]) * 1e12
final_cost_m2 = raw_cost_m2 * pressure_factor(t_m2_real, T_MIN, T_MAX)
cost_m2_tr = final_cost_m2 / 1e12
# 曲线终点校准到含罚款的终点（保持展示一致）
traj_m2 = m2_costs_tr * (final_cost_m2 / raw_cost_m2) if raw_cost_m2 > 0 else m2_costs_tr
years_m2 = START_YEAR + m2_days

# M3：Hybrid（计划 150 年；先规划再按计划配速执行火箭；不允许提前完成）
t_plan_m3 = 150.0
_, m_roc_plan, m_ele_plan, _ = solve_allocation_planning(t_plan_m3, M_TOTAL)
print(f"  Simulating M3 hybrid: rockets={m_roc_plan/1e6:.1f} Mt, elevator={m_ele_plan/1e6:.1f} Mt")

# 火箭执行：按 150 年配速节流
m3_roc_days, m3_roc_costs_tr, _ = simulate_rocket_trajectory(m_roc_plan, seed=SOFT_SEED, pace_years=t_plan_m3)
t_roc_m3 = float(m3_roc_days[-1])
raw_cost_roc_m3 = float(m3_roc_costs_tr[-1]) * 1e12

# 电梯：理想线性跑完计划分配量
t_ele_m3 = (m_ele_plan / yearly_payload_g) if yearly_payload_g > 0 else 0.0
ele_cost_total_tr = (m_ele_plan * cost_g) / 1e12

# 真实交付时间（不允许提前完成，至少到 150）
t_real_m3 = max(t_plan_m3, t_ele_m3, t_roc_m3)

# 统一时间轴合并成本
common_t = np.linspace(0.0, t_real_m3, 500)

interp_roc = np.interp(common_t, m3_roc_days, m3_roc_costs_tr, right=m3_roc_costs_tr[-1])
# 电梯成本：线性增长到 t_ele_m3（若 t_ele_m3 < t_plan_m3 也无所谓，仍保持完成后持平）
if t_ele_m3 > 0:
    ele_curve = np.minimum(common_t / t_ele_m3, 1.0) * ele_cost_total_tr
else:
    ele_curve = np.zeros_like(common_t)

raw_total_m3_tr = interp_roc + ele_curve
raw_total_m3_usd = float(raw_total_m3_tr[-1]) * 1e12

final_cost_m3_usd = raw_total_m3_usd * pressure_factor(t_real_m3, T_MIN, T_MAX)
cost_m3_tr = final_cost_m3_usd / 1e12

# 曲线终点校准到含罚款终点
scale_m3 = (final_cost_m3_usd / raw_total_m3_usd) if raw_total_m3_usd > 0 else 1.0
traj_m3 = raw_total_m3_tr * scale_m3
years_m3 = START_YEAR + common_t

# points
df_pareto_points = pd.DataFrame([
    {"scenario": SCENARIO, "method": "M1", "time_years": t_m1, "cost_trillion": cost_m1_tr, "time_kind": "planned", "note": "Elevator only (ideal)"},
    {"scenario": SCENARIO, "method": "M2", "time_years": t_m2_real, "cost_trillion": cost_m2_tr, "time_kind": "real", "note": f"Rockets stochastic (seed={SOFT_SEED})"},
    {"scenario": SCENARIO, "method": "M3", "time_years": t_real_m3, "cost_trillion": cost_m3_tr, "time_kind": "real", "note": f"Hybrid stochastic (seed={SOFT_SEED}), no-early-finish"},
])

# traj series
df_traj_series = pd.concat([
    pd.DataFrame({"scenario": SCENARIO, "curve": "M1", "x_year": years_m1, "y_cost_trillion": traj_m1, "seed": np.nan, "note": "Ideal elevator"}),
    pd.DataFrame({"scenario": SCENARIO, "curve": "M2", "x_year": years_m2, "y_cost_trillion": traj_m2, "seed": SOFT_SEED, "note": "Stochastic rockets"}),
    pd.DataFrame({"scenario": SCENARIO, "curve": "M3", "x_year": years_m3, "y_cost_trillion": traj_m3, "seed": SOFT_SEED, "note": "Hybrid stochastic"}),
], ignore_index=True)

df_traj_band = pd.DataFrame(columns=[
    "scenario", "band", "x_year", "y_lower_trillion", "y_upper_trillion", "level"
])

# =============================================================================
# 9. meta & Excel 导出
# =============================================================================
df_meta = pd.DataFrame([
    ("scenario", SCENARIO),
    ("RGEO", RGEO),
    ("SCALE_DISCOUNT", SCALE_DISCOUNT),
    ("START_YEAR", START_YEAR),
    ("M_TOTAL", M_TOTAL),
    ("T_MIN_ref", T_MIN),
    ("T_MAX_ref", T_MAX),
    ("SOFT_SEED", SOFT_SEED),
    ("note", "Planning optimistic (ideal capacity). Execution: stochastic rocket base failures; t_real=max(T_plan,t_ele,t_roc) no-early-finish; pressure=max(0,..)."),
], columns=["key", "value"])

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
# 10. 绘图（可选）
# =============================================================================
if PLOT:
    # Pareto
    plt.figure(figsize=(12, 7))

    df_plan = df_pareto_series[df_pareto_series["curve"] == "base_pareto_planning"]
    df_soft = df_pareto_series[df_pareto_series["curve"] == "soft_deadline_single_seed"]

    plt.plot(df_plan["x_time_years"], df_plan["y_cost_trillion"], color=colors["hybrid"], linewidth=3,
             label="Hybrid Pareto (Planning, optimistic)")
    plt.fill_between(df_plan["x_time_years"], df_plan["y_cost_trillion"], color=colors["hybrid"], alpha=0.10)

    plt.plot(df_soft["x_time_years"], df_soft["y_cost_trillion"], 'k--', linewidth=3,
             label=f"Soft Deadline Pareto (Execution, seed={SOFT_SEED})")
    plt.fill_between(df_soft["x_time_years"], df_soft["y_cost_trillion"], color="black", alpha=0.06)

    # points
    for _, r in df_pareto_points.iterrows():
        c = {"M1": colors["elevator"], "M2": colors["rocket"], "M3": colors["hybrid"]}[r["method"]]
        plt.scatter(r["time_years"], r["cost_trillion"], color=c, s=120, zorder=5, edgecolors='black')
        plt.annotate(f'{r["method"]}\n({r["time_years"]:.1f}y, ${r["cost_trillion"]:.1f}T)',
                     (r["time_years"], r["cost_trillion"]),
                     xytext=(10, 10), textcoords='offset points',
                     color=c, fontweight='bold', fontsize=9)

    plt.title("Pareto Frontier: Impact of Base Reliability (Aligned)", fontsize=14)
    plt.xlabel("Completion Time (Years)")
    plt.ylabel("Total Cost (Trillion USD)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    # Trajectory
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(years_m1, traj_m1, color=colors["elevator"], linewidth=2.5, label="M1: Elevator (Ideal)")
    ax.plot(years_m2, traj_m2, color=colors["rocket"], linewidth=2.5, label=f"M2: Rockets (Stochastic, seed={SOFT_SEED})")
    ax.plot(years_m3, traj_m3, color=colors["hybrid"], linewidth=2.5, label=f"M3: Hybrid (Stochastic, seed={SOFT_SEED})")

    ax.set_title("Cumulative Cost Trajectories (Aligned)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Year")
    ax.set_ylabel("Accumulated Cost (Trillion USD)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(START_YEAR, START_YEAR + 450)
    ax.legend(loc="upper left", frameon=False)

    axins = inset_axes(ax, width="40%", height="40%", loc=1, borderpad=3)
    axins.plot(years_m1, traj_m1, color=colors["elevator"])
    axins.plot(years_m2, traj_m2, color=colors["rocket"])
    axins.plot(years_m3, traj_m3, color=colors["hybrid"])
    axins.set_xlim(2050, 2220)
    axins.set_ylim(0, max(cost_m2_tr, cost_m3_tr, cost_m1_tr) * 1.1)
    axins.grid(True, linestyle=":", alpha=0.4)
    axins.set_title("Zoom: 2050-2220", fontsize=9)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--")

    plt.tight_layout()
    plt.show()

print("--- Code3 aligned run complete ---")