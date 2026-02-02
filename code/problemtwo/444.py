import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# =============================================================================
# 0. 开关
# =============================================================================
PLOT = True
EXPORT_EXCEL = True
EXCEL_PATH = "plot_data_code4_launch_failure.xlsx"

SCENARIO = "s4_launch_failure"

# =============================================================================
# 1. 基础参数与设置（RGEO 以代码五为准）
# =============================================================================
M_TOTAL = 1e8
G0, MU, RE, RGEO = 9.80665, 398600.44, 6378.0, 106378.0  # <-- aligned to code5
V_ROT_EQUATOR = 0.4651
ALPHA_G, ALPHA_E = 0.3, 0.6
L_MAX = 6000.0
DV_G, ISP_G = 2.2, 460.0
DV_BASE, ISP_E = 15.2, 430.0
C_DRY, C_PROP = 3.1e6, 500.0
N_PORTS, U_PORT = 3, 179000.0
ETA_E, PI_E = 0.8, 0.03
LAMBDA_J = 2.0
SCALE_DISCOUNT = 0.2  # 折扣上限固定 20%
START_YEAR = 2050

# --- Failure model params (你要求：每天每基地一次失败抽样) ---
P_FAIL_CONST = 0.01          # 每次尝试失败概率（固定 1%）
C_PAYLOAD_PER_TON = 1e5      # 货物价值（失败时损失）$100,000/ton

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
# 2. 通用工具函数（压力项统一 + 折扣封顶）
# =============================================================================
def clamp_discount(d):
    return float(min(SCALE_DISCOUNT, max(0.0, d)))

def pressure_factor(t_used, t_min_ref, t_max_ref):
    """
    统一压力项：只惩罚“赶工”（更快更贵），不惩罚更慢：
    pressure = max(0, (T_MAX - t_used)/(T_MAX - T_MIN))
    """
    denom = (t_max_ref - t_min_ref)
    if denom <= 0:
        return 1.0
    p = (t_max_ref - t_used) / denom
    p = max(0.0, p)
    return 1.0 + 0.5 * (p ** 2)

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
total_yearly_rockets = float(df_bases["Yearly"].sum())

# 参考极限（规划层/压力项基准仍用理想产能）
T_MIN = M_TOTAL / (yearly_payload_g + total_yearly_rockets)
T_MAX = M_TOTAL / yearly_payload_g

# =============================================================================
# 4. 规划层：乐观分配（不预知失败）
# =============================================================================
def solve_allocation_with_integral_cost(time_target, total_mass):
    final_marginal_discount = 0.0

    for _ in range(10):
        avg_discount = clamp_discount(final_marginal_discount * 0.5)
        avg_rocket_cost = float((df_bases["Cost"] * (1 - avg_discount)).mean())

        rem = total_mass
        cost_iter = 0.0
        mass_ele = 0.0
        mass_roc = 0.0

        if avg_rocket_cost < cost_g:
            can_take_roc = min(rem, total_yearly_rockets * time_target)
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
                can_take = min(needed_rocket, b["Yearly"] * time_target)
                cost_iter += can_take * b["Cost"] * (1 - avg_discount)
                mass_roc += can_take
                needed_rocket -= can_take

        new_marginal_discount = clamp_discount(SCALE_DISCOUNT * (mass_roc / total_mass))
        if abs(new_marginal_discount - final_marginal_discount) < 1e-4:
            return cost_iter, mass_roc, mass_ele, new_marginal_discount

        final_marginal_discount = new_marginal_discount

    return cost_iter, mass_roc, mass_ele, final_marginal_discount

# =============================================================================
# 5. 执行层：火箭“尝试-失败”离散过程（失败也推进学习：attempted_mass）
# =============================================================================
def simulate_rocket_trajectory_with_failure(
    target_delivered_mass,
    seed=42,
    pace_years=None,
    discount_base_mass=None
):
    """
    核心口径（你要求）：
    - 每天每基地一次“尝试”抽样（如果 prod>0 就算一次尝试）
    - 失败：不增加 delivered mass，但增加成本（发射损失+货物损失）
    - 学习曲线进度变量：attempted_mass（失败也算尝试推进学习）
    - 折扣上限 clamp 到 20%
    - pace_years：按计划配速节流（不补偿失败、不加班）
    """
    if target_delivered_mass <= 0:
        return np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])

    rng = np.random.default_rng(seed)

    daily_caps = df_bases["Yearly"].values / 365.0
    unit_costs = df_bases["Cost"].values
    num_bases = len(df_bases)

    # 节流：按计划配速（对 delivered_mass 的目标配速）
    if pace_years is not None:
        daily_needed = target_delivered_mass / (pace_years * 365.0)
        daily_max = float(np.sum(daily_caps))
        if daily_max > daily_needed:
            throttle = daily_needed / daily_max
            daily_caps = daily_caps * throttle

    if discount_base_mass is None:
        discount_base_mass = M_TOTAL

    # 记录：days / cost / delivered / attempted
    days = [0.0]
    costs = [0.0]
    delivered = [0.0]
    attempted = [0.0]

    delivered_mass = 0.0
    attempted_mass = 0.0
    total_cost = 0.0
    day_count = 0

    # 安全阈值防死循环
    MAX_DAYS = 2000 * 365

    while delivered_mass < target_delivered_mass and day_count < MAX_DAYS:
        day_count += 1
        delivered_today = 0.0

        for i in range(num_bases):
            prod = float(daily_caps[i])

            # delivered 端截断：防止超发
            if delivered_mass + delivered_today + prod > target_delivered_mass:
                prod = target_delivered_mass - (delivered_mass + delivered_today)

            if prod <= 0:
                continue

            # --- 每基地一次尝试：无论成功失败都计入 attempted_mass ---
            # attempted 用 midpoint 近似也可以，但你要求“失败也推进学习”，所以只要加 prod 即可
            attempted_mid = attempted_mass + prod / 2.0
            discount_now = clamp_discount(SCALE_DISCOUNT * (attempted_mid / discount_base_mass))

            base_launch_cost = prod * unit_costs[i]

            if rng.random() < P_FAIL_CONST:
                # 失败：发射成本(折扣后) + 货物损失(全额)
                total_cost += base_launch_cost * (1 - discount_now)
                total_cost += prod * C_PAYLOAD_PER_TON
                attempted_mass += prod
                # delivered 不变
                continue
            else:
                # 成功：发射成本(折扣后)，delivered 增加
                total_cost += base_launch_cost * (1 - discount_now)
                attempted_mass += prod
                delivered_today += prod

        delivered_mass += delivered_today

        if (day_count % 30 == 0) or (delivered_mass >= target_delivered_mass):
            days.append(day_count / 365.0)
            costs.append(total_cost / 1e12)
            delivered.append(delivered_mass)
            attempted.append(attempted_mass)

    return (
        np.array(days),
        np.array(costs),
        np.array(delivered),
        np.array(attempted)
    )

# =============================================================================
# 6. True-process Pareto evaluator（计划 T -> 执行得 t_real & cost_real）
# =============================================================================
def evaluate_hybrid_plan_true_process(time_plan, total_mass, seed=101):
    """
    规划：仍用乐观求解器（不预知失败）
    执行：
      - 电梯按计划线性完成 m_ele_plan，完工时间 t_ele = m_ele_plan / yearly_payload_g
      - 火箭按计划配速节流尝试，直到 delivered 完成 m_roc_plan，得到 t_roc
    交付口径：不允许提前完成 -> t_real = max(time_plan, t_ele, t_roc)
    """
    _, m_roc_plan, m_ele_plan, _ = solve_allocation_with_integral_cost(time_plan, total_mass)

    # 电梯：不受炸机影响
    t_ele = (m_ele_plan / yearly_payload_g) if yearly_payload_g > 0 else 0.0
    cost_ele_usd = m_ele_plan * cost_g

    # 火箭：带失败，按 time_plan 配速
    roc_days, roc_costs_tr, _, _ = simulate_rocket_trajectory_with_failure(
        m_roc_plan,
        seed=seed,
        pace_years=time_plan,
        discount_base_mass=M_TOTAL  # 统一基准
    )
    t_roc = float(roc_days[-1])
    cost_roc_usd = float(roc_costs_tr[-1]) * 1e12

    t_real = max(time_plan, t_ele, t_roc)
    raw_cost_usd = cost_ele_usd + cost_roc_usd
    final_cost_usd = raw_cost_usd * pressure_factor(t_real, T_MIN, T_MAX)

    return t_real, final_cost_usd / 1e12

# =============================================================================
# 7. 数据生成：Pareto（planning 基底线 + true-process 线）
# =============================================================================
print("Running scenario 4 (Launch failure) aligned...")

t_axis = np.linspace(T_MIN, T_MAX, 100)

# A) 基底 planning Pareto（不含失败）
pareto_plan_rows = []
for T in t_axis:
    raw_cost, _, _, _ = solve_allocation_with_integral_cost(T, M_TOTAL)
    final_cost = raw_cost * pressure_factor(T, T_MIN, T_MAX)
    pareto_plan_rows.append({
        "scenario": SCENARIO,
        "curve": "base_pareto_planning",
        "x_time_years": T,
        "y_cost_trillion": final_cost / 1e12,
        "x_kind": "planned",
        "seed": np.nan,
        "note": "Planning optimistic (no failure)"
    })

# B) True-process Pareto（带失败仿真，单种子随 T 变化）
pareto_true_rows = []
print("  Building Pareto (True Process w/ Failure) ...")
for idx, T in enumerate(t_axis):
    seed_T = 1000 + idx  # 每个点一个 seed，避免整条线完全同噪声形态
    t_real, cost_tr = evaluate_hybrid_plan_true_process(T, M_TOTAL, seed=seed_T)
    pareto_true_rows.append({
        "scenario": SCENARIO,
        "curve": "true_process_failure",
        "x_time_years": t_real,       # 横轴按真实完工时间
        "y_cost_trillion": cost_tr,
        "x_kind": "real",
        "seed": seed_T,
        "note": "Execution stochastic failure; no-early-finish"
    })

df_pareto_series = pd.DataFrame(pareto_plan_rows + pareto_true_rows)
df_pareto_band = pd.DataFrame(columns=[
    "scenario", "band", "x_time_years", "y_lower_trillion", "y_upper_trillion", "level", "stat_center"
])

# =============================================================================
# 8. 关键点与累计成本曲线（M1/M2/M3）
# =============================================================================
# M1：电梯 only（理想）
t_m1 = T_MAX
cost_m1_tr = (M_TOTAL * cost_g) / 1e12
years_m1 = np.linspace(START_YEAR, START_YEAR + t_m1, 200)
traj_m1 = np.linspace(0.0, cost_m1_tr, 200)

# M2：火箭 only（带失败，全速跑）
print("  Simulating M2 (Rocket w/ Failure) ...")
m2_days, m2_costs_tr, m2_delivered, m2_attempted = simulate_rocket_trajectory_with_failure(
    M_TOTAL, seed=101, pace_years=None, discount_base_mass=M_TOTAL
)
t_m2_real = float(m2_days[-1])
raw_cost_m2 = float(m2_costs_tr[-1]) * 1e12
final_cost_m2 = raw_cost_m2 * pressure_factor(t_m2_real, T_MIN, T_MAX)
cost_m2_tr = final_cost_m2 / 1e12
# 曲线终点校准到含罚款终点
traj_m2 = m2_costs_tr * (final_cost_m2 / raw_cost_m2) if raw_cost_m2 > 0 else m2_costs_tr
years_m2 = START_YEAR + m2_days

# M3：Hybrid（计划 150 年；规划乐观；执行火箭按 150 年配速；不允许提前完成）
print("  Simulating M3 (Hybrid w/ Failure) ...")
t_plan_m3 = 150.0
_, m_roc_plan, m_ele_plan, _ = solve_allocation_with_integral_cost(t_plan_m3, M_TOTAL)

t_ele_m3 = (m_ele_plan / yearly_payload_g) if yearly_payload_g > 0 else 0.0
ele_cost_total_tr = (m_ele_plan * cost_g) / 1e12

m3_roc_days, m3_roc_costs_tr, _, _ = simulate_rocket_trajectory_with_failure(
    m_roc_plan, seed=101, pace_years=t_plan_m3, discount_base_mass=M_TOTAL
)
t_roc_m3 = float(m3_roc_days[-1])

# 交付口径：不允许提前完成
t_real_m3 = max(t_plan_m3, t_ele_m3, t_roc_m3)

# 合并成本曲线（统一时间轴）
common_t = np.linspace(0.0, t_real_m3, 1000)
interp_roc = np.interp(common_t, m3_roc_days, m3_roc_costs_tr, right=m3_roc_costs_tr[-1])

if t_ele_m3 > 0:
    ele_curve = np.minimum(common_t / t_ele_m3, 1.0) * ele_cost_total_tr
else:
    ele_curve = np.zeros_like(common_t)

raw_total_m3_tr = interp_roc + ele_curve
raw_total_m3_usd = float(raw_total_m3_tr[-1]) * 1e12
final_cost_m3_usd = raw_total_m3_usd * pressure_factor(t_real_m3, T_MIN, T_MAX)
cost_m3_tr = final_cost_m3_usd / 1e12

scale_m3 = (final_cost_m3_usd / raw_total_m3_usd) if raw_total_m3_usd > 0 else 1.0
traj_m3 = raw_total_m3_tr * scale_m3
years_m3 = START_YEAR + common_t

# points
df_pareto_points = pd.DataFrame([
    {"scenario": SCENARIO, "method": "M1", "time_years": t_m1, "cost_trillion": cost_m1_tr, "time_kind": "planned", "note": "Elevator only (ideal)"},
    {"scenario": SCENARIO, "method": "M2", "time_years": t_m2_real, "cost_trillion": cost_m2_tr, "time_kind": "real", "note": "Rocket only w/ failure (seed=101)"},
    {"scenario": SCENARIO, "method": "M3", "time_years": t_real_m3, "cost_trillion": cost_m3_tr, "time_kind": "real", "note": "Hybrid w/ failure (seed=101), no-early-finish"},
])

# traj series
df_traj_series = pd.concat([
    pd.DataFrame({"scenario": SCENARIO, "curve": "M1", "x_year": years_m1, "y_cost_trillion": traj_m1, "seed": np.nan, "note": "Ideal elevator"}),
    pd.DataFrame({"scenario": SCENARIO, "curve": "M2", "x_year": years_m2, "y_cost_trillion": traj_m2, "seed": 101, "note": "Rocket failure stochastic"}),
    pd.DataFrame({"scenario": SCENARIO, "curve": "M3", "x_year": years_m3, "y_cost_trillion": traj_m3, "seed": 101, "note": "Hybrid failure stochastic"}),
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
    ("P_FAIL_CONST", P_FAIL_CONST),
    ("C_PAYLOAD_PER_TON", C_PAYLOAD_PER_TON),
    ("note", "Failure model: daily per-base Bernoulli; learning uses attempted_mass (failures accelerate learning), discount capped 20%; planning optimistic; t_real=max(T_plan,t_ele,t_roc) no-early-finish; pressure=max(0,..)."),
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
    df_true = df_pareto_series[df_pareto_series["curve"] == "true_process_failure"]

    plt.plot(df_plan["x_time_years"], df_plan["y_cost_trillion"], color=colors["hybrid"], linewidth=3,
             label="Hybrid Pareto (Planning, no failure)")
    plt.fill_between(df_plan["x_time_years"], df_plan["y_cost_trillion"], color=colors["hybrid"], alpha=0.10)

    plt.plot(df_true["x_time_years"], df_true["y_cost_trillion"], color="orange", linewidth=3,
             label="Hybrid Pareto (True Process w/ Failure)")
    plt.fill_between(df_true["x_time_years"], df_true["y_cost_trillion"], color="orange", alpha=0.10)

    # points
    for _, r in df_pareto_points.iterrows():
        c = {"M1": colors["elevator"], "M2": colors["rocket"], "M3": colors["hybrid"]}[r["method"]]
        plt.scatter(r["time_years"], r["cost_trillion"], color=c, s=120, zorder=5, edgecolors='black')
        plt.annotate(f'{r["method"]}\n({r["time_years"]:.1f}y, ${r["cost_trillion"]:.1f}T)',
                     (r["time_years"], r["cost_trillion"]),
                     xytext=(10, 10), textcoords='offset points',
                     color=c, fontweight='bold', fontsize=9)

    plt.title("Pareto Frontier: Launch Failure Risk (Aligned)", fontsize=14)
    plt.xlabel("Completion Time (Years)")
    plt.ylabel("Total Cost (Trillion USD)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    # Trajectory
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(years_m1, traj_m1, color=colors["elevator"], linewidth=2.5, label="M1: Elevator (Ideal)")
    ax.plot(years_m2, traj_m2, color=colors["rocket"], linewidth=2.5, label="M2: Rockets (Failure, seed=101)")
    ax.plot(years_m3, traj_m3, color=colors["hybrid"], linewidth=2.5, label="M3: Hybrid (Failure, seed=101)")

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

print("--- Code4 aligned run complete ---")