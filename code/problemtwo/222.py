import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# =============================================================================
# 0. 开关
# =============================================================================
PLOT = True
EXPORT_EXCEL = True
EXCEL_PATH = "plot_data_code2_sway.xlsx"

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
SCALE_DISCOUNT = 0.2  # 20% 规模折扣（封顶）
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

SCENARIO = "s2_sway"

# =============================================================================
# 2. 通用工具函数（统一压力项口径：max(0, …)）
# =============================================================================
def clamp_discount(d):
    return float(min(SCALE_DISCOUNT, max(0.0, d)))

def pressure_factor(t_used, t_min_ref, t_max_ref):
    """
    统一压力项：只惩罚“赶工”（更快更贵），不惩罚更慢。
    pressure = max(0, (T_MAX - t_used)/(T_MAX - T_MIN))
    """
    denom = (t_max_ref - t_min_ref)
    if denom <= 0:
        return 1.0
    p = (t_max_ref - t_used) / denom
    p = max(0.0, p)
    return 1.0 + 0.5 * (p ** 2)

# =============================================================================
# 3. 蒙特卡洛采样：截断正态分布（局部 RNG）
# =============================================================================
def generate_monte_carlo_samples(rng, n_samples=5000, mean=1.05, std=0.05, lower_bound=1.0):
    """
    生成截断正态样本 Y (Y>=lower_bound)，Y 越大电梯运力越低（位于分母）。
    使用局部 rng，避免污染全局随机数状态。
    """
    raw_count = int(n_samples * 1.5)
    samples = []

    while len(samples) < n_samples:
        raw = rng.normal(mean, std, raw_count)
        valid = raw[raw >= lower_bound]
        samples.extend(valid.tolist())

    return np.array(samples[:n_samples], dtype=float)

# =============================================================================
# 4. 核心计算函数
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
yearly_payload_g_nominal = (N_PORTS * U_PORT) / k_g  # Y=1 的标称电梯年运力

base_list = []
for name, lat in BASES_DATA.items():
    v_rot = V_ROT_EQUATOR * np.cos(np.radians(lat))
    dv_eff = DV_BASE - v_rot
    k, c = get_rocket_stats(dv_eff, ISP_E, ALPHA_E, False)
    max_y = (LAMBDA_J * 365 * L_MAX) / k
    base_list.append({"Cost": c, "Yearly": max_y, "Name": name})

df_bases = pd.DataFrame(base_list).sort_values("Cost").reset_index(drop=True)
total_yearly_rockets = df_bases["Yearly"].sum()

# 统一用“标称”作为参考 T_MIN / T_MAX（压力项基准也用它）
T_MIN_NOMINAL = M_TOTAL / (yearly_payload_g_nominal + total_yearly_rockets)
T_MAX_NOMINAL = M_TOTAL / yearly_payload_g_nominal

# =============================================================================
# 5. 规划求解器：电梯运力受 sway_Y 影响（规划层对 Y “已知”，做条件分析）
# =============================================================================
def solve_allocation_with_integral_cost(time_target, total_mass, sway_Y=1.0):
    """
    规划层求解：电梯年运力 = 标称 / Y
    学习曲线：规划层无失败，attempted≈delivered=m_roc，基准=全项目 M_TOTAL，封顶 20%
    """
    current_yearly_elevator = yearly_payload_g_nominal / sway_Y

    final_marginal_discount = 0.0
    for _ in range(10):
        avg_discount = clamp_discount(final_marginal_discount * 0.5)

        avg_rocket_cost = float((df_bases["Cost"] * (1 - avg_discount)).mean())

        rem = total_mass
        cost_iter = 0.0
        mass_ele = 0.0
        mass_roc = 0.0

        if avg_rocket_cost < cost_g:
            # 极端：火箭更便宜 -> 优先火箭
            can_take_roc = min(rem, total_yearly_rockets * time_target)
            cost_iter += can_take_roc * avg_rocket_cost
            mass_roc += can_take_roc
            rem -= can_take_roc

            cost_iter += rem * cost_g
            mass_ele += rem
        else:
            # 正常：优先电梯（受 sway 影响）
            can_take_ele = min(rem, current_yearly_elevator * time_target)
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
# 6. 生成 Monte Carlo：Pareto band + Trajectory band
# =============================================================================
print("Running scenario 2 (Sway uncertainty) aligned...")

# 随机种子策略（可复现）
MASTER_SEED = 20260202
rng = np.random.default_rng(MASTER_SEED)

N_SAMPLES = 5000
sway_samples = generate_monte_carlo_samples(rng, n_samples=N_SAMPLES, mean=1.05, std=0.05, lower_bound=1.0)

# t_axis：为了容纳更慢的情况，延长到 1.5*T_MAX_NOMINAL
t_axis = np.linspace(T_MIN_NOMINAL, T_MAX_NOMINAL * 1.5, 100)

# --- Method 2：不受 sway 影响（确定性点/线）
t_m2 = M_TOTAL / total_yearly_rockets
avg_discount_m2 = clamp_discount(SCALE_DISCOUNT * 0.5)
c_m2_raw = 0.0
for _, b in df_bases.iterrows():
    c_m2_raw += b["Yearly"] * t_m2 * b["Cost"]
c_m2_raw *= (1 - avg_discount_m2)

final_cost_m2 = c_m2_raw * pressure_factor(t_m2, T_MIN_NOMINAL, T_MAX_NOMINAL)
cost_m2_tr = final_cost_m2 / 1e12

# --- Pareto：对每个 Y，计算一条 Hybrid 成本曲线（不可行点记 NaN）
costs_matrix = np.full((N_SAMPLES, len(t_axis)), np.nan, dtype=float)

# 同时记录 M1（电梯 only）的时间分布与 cost（cost 不变：按吨收费）
m1_times = np.zeros(N_SAMPLES, dtype=float)
m1_cost_tr = (M_TOTAL * cost_g) / 1e12  # 不随 Y 变（按吨计费）

for si, Y_val in enumerate(sway_samples):
    real_ele_cap = yearly_payload_g_nominal / Y_val
    m1_times[si] = M_TOTAL / real_ele_cap

    for ti, T in enumerate(t_axis):
        # 可行性：在该 Y 下，T 年最多完成的总运量
        max_possible = (real_ele_cap + total_yearly_rockets) * T
        if max_possible < M_TOTAL:
            continue  # keep NaN

        raw_cost, _, _, _ = solve_allocation_with_integral_cost(T, M_TOTAL, sway_Y=Y_val)

        # 统一压力项：以“标称”参考为基准（你的原代码也是这个思路），并做 max(0) 截断
        final_cost = raw_cost * pressure_factor(T, T_MIN_NOMINAL, T_MAX_NOMINAL)
        costs_matrix[si, ti] = final_cost / 1e12

# 统计：Hybrid mean & 5/95
mean_cost = np.nanmean(costs_matrix, axis=0)
lower_cost = np.nanpercentile(costs_matrix, 5, axis=0)
upper_cost = np.nanpercentile(costs_matrix, 95, axis=0)

# 对齐：剔除全 NaN 的点
valid_mask = ~np.isnan(mean_cost)

df_pareto_series = pd.DataFrame({
    "scenario": SCENARIO,
    "curve": "hybrid_mean",
    "x_time_years": t_axis[valid_mask],
    "y_cost_trillion": mean_cost[valid_mask],
    "x_kind": "planned",
    "seed": np.nan,
    "note": f"MonteCarlo mean, N={N_SAMPLES}"
})

df_pareto_band = pd.DataFrame({
    "scenario": SCENARIO,
    "band": "hybrid_ci95",
    "x_time_years": t_axis[valid_mask],
    "y_lower_trillion": lower_cost[valid_mask],
    "y_upper_trillion": upper_cost[valid_mask],
    "level": 0.95,
    "stat_center": "mean"
})

# Pareto points：M1 nominal, M2 robust, 以及 M3 nominal@150（Y=1）
t_m1_nom = T_MAX_NOMINAL
cost_m1_nom = (M_TOTAL * cost_g) / 1e12

t_m3_nom = 150.0
raw_m3_nom, m_roc_m3_nom, m_ele_m3_nom, _ = solve_allocation_with_integral_cost(t_m3_nom, M_TOTAL, sway_Y=1.0)
final_m3_nom = raw_m3_nom * pressure_factor(t_m3_nom, T_MIN_NOMINAL, T_MAX_NOMINAL)
cost_m3_nom = final_m3_nom / 1e12

df_pareto_points = pd.DataFrame([
    {"scenario": SCENARIO, "method": "M1", "time_years": t_m1_nom, "cost_trillion": cost_m1_nom, "time_kind": "planned", "note": "Nominal (Y=1)"},
    {"scenario": SCENARIO, "method": "M2", "time_years": t_m2, "cost_trillion": cost_m2_tr, "time_kind": "planned", "note": "Deterministic (unaffected by sway)"},
    {"scenario": SCENARIO, "method": "M3", "time_years": t_m3_nom, "cost_trillion": cost_m3_nom, "time_kind": "planned", "note": "Nominal Hybrid @150, Y=1"},
])

# =============================================================================
# 7. Trajectory（累计成本曲线）：输出 mean + CI band（M1/M3），M2 为确定性线
# =============================================================================
def get_curve(duration, total_cost_usd, m_ele, m_roc):
    """
    画图用“形状示意 + 终点校准”的曲线：
    - 电梯与火箭按分配的总量匀速推进
    - 火箭用平均基地单价 + 学习折扣（规划层：attempted≈delivered）
    - 最后 scale 到 total_cost_usd（保证终点一致）
    """
    years = np.linspace(0, duration, 200)

    step_ele = m_ele / 200.0
    step_roc = m_roc / 200.0

    total_cap = df_bases["Yearly"].sum()
    avg_base_price = float((df_bases["Cost"] * (df_bases["Yearly"] / total_cap)).sum())

    cumulative_cost = 0.0
    cumulative_roc_mass = 0.0  # 无失败：attempted=delivered

    costs_tr = []
    for _ in years:
        costs_tr.append(cumulative_cost / 1e12)

        cost_step_ele = step_ele * cost_g

        progress = (cumulative_roc_mass / M_TOTAL) if M_TOTAL > 0 else 0.0
        discount = clamp_discount(SCALE_DISCOUNT * progress)
        cost_step_roc = step_roc * (avg_base_price * (1 - discount))

        cumulative_cost += cost_step_ele + cost_step_roc
        cumulative_roc_mass += step_roc

    scale_fix = (total_cost_usd / cumulative_cost) if cumulative_cost > 0 else 1.0
    return years + START_YEAR, np.array(costs_tr) * scale_fix

# 统一的 year 轴（用于插值统计 band）
COMMON_YEARS = np.linspace(START_YEAR, START_YEAR + 450, 500)

# --- M3：固定计划时间 150 年下，Y 不确定导致分配与总成本不确定
t_target = 150.0

m3_curves = []
for Y_val in sway_samples:
    # 如果在该 Y 下 150 年不可行，则这条曲线全 NaN
    real_ele_cap = yearly_payload_g_nominal / Y_val
    if (real_ele_cap + total_yearly_rockets) * t_target < M_TOTAL:
        m3_curves.append(np.full_like(COMMON_YEARS, np.nan, dtype=float))
        continue

    raw, m_roc, m_ele, _ = solve_allocation_with_integral_cost(t_target, M_TOTAL, sway_Y=Y_val)
    total_cost_usd = raw * pressure_factor(t_target, T_MIN_NOMINAL, T_MAX_NOMINAL)

    ys, cs = get_curve(t_target, total_cost_usd, m_ele, m_roc)
    interp = np.interp(COMMON_YEARS, ys, cs, right=np.nan)
    m3_curves.append(interp)

m3_arr = np.array(m3_curves, dtype=float)
m3_mean = np.nanmean(m3_arr, axis=0)
m3_lo = np.nanpercentile(m3_arr, 5, axis=0)
m3_hi = np.nanpercentile(m3_arr, 95, axis=0)

df_traj_series_m3 = pd.DataFrame({
    "scenario": SCENARIO,
    "curve": "M3_mean",
    "x_year": COMMON_YEARS,
    "y_cost_trillion": m3_mean,
    "seed": np.nan,
    "note": f"Mean trajectory @T=150, N={N_SAMPLES}"
})

df_traj_band_m3 = pd.DataFrame({
    "scenario": SCENARIO,
    "band": "M3_ci95",
    "x_year": COMMON_YEARS,
    "y_lower_trillion": m3_lo,
    "y_upper_trillion": m3_hi,
    "level": 0.95
})

# --- M1：电梯 only，时间不确定，但成本总额不变；曲线用“按实际工期线性到达终点”再统计 band
m1_curves = []
for t_finish in m1_times:
    ys = np.linspace(START_YEAR, START_YEAR + t_finish, 200)
    cs = np.linspace(0.0, m1_cost_tr, 200)
    interp = np.interp(COMMON_YEARS, ys, cs, right=np.nan)
    m1_curves.append(interp)

m1_arr = np.array(m1_curves, dtype=float)
m1_mean = np.nanmean(m1_arr, axis=0)
m1_lo = np.nanpercentile(m1_arr, 5, axis=0)
m1_hi = np.nanpercentile(m1_arr, 95, axis=0)

df_traj_series_m1 = pd.DataFrame({
    "scenario": SCENARIO,
    "curve": "M1_mean",
    "x_year": COMMON_YEARS,
    "y_cost_trillion": m1_mean,
    "seed": np.nan,
    "note": f"Mean trajectory, N={N_SAMPLES}"
})

df_traj_band_m1 = pd.DataFrame({
    "scenario": SCENARIO,
    "band": "M1_ci95",
    "x_year": COMMON_YEARS,
    "y_lower_trillion": m1_lo,
    "y_upper_trillion": m1_hi,
    "level": 0.95
})

# --- M2：确定性轨迹（用基底式“形状示意 + 终点校准”即可）
ys_m2, cs_m2 = get_curve(t_m2, final_cost_m2, 0.0, M_TOTAL)
df_traj_series_m2 = pd.DataFrame({
    "scenario": SCENARIO,
    "curve": "M2",
    "x_year": ys_m2,
    "y_cost_trillion": cs_m2,
    "seed": np.nan,
    "note": "Deterministic"
})

df_traj_series = pd.concat([df_traj_series_m1, df_traj_series_m2, df_traj_series_m3], ignore_index=True)
df_traj_band = pd.concat([df_traj_band_m1, df_traj_band_m3], ignore_index=True)

# =============================================================================
# 8. meta_params & Excel 导出（统一 schema）
# =============================================================================
df_meta = pd.DataFrame([
    ("scenario", SCENARIO),
    ("RGEO", RGEO),
    ("SCALE_DISCOUNT", SCALE_DISCOUNT),
    ("START_YEAR", START_YEAR),
    ("M_TOTAL", M_TOTAL),
    ("T_MIN_NOMINAL", T_MIN_NOMINAL),
    ("T_MAX_NOMINAL", T_MAX_NOMINAL),
    ("N_SAMPLES", N_SAMPLES),
    ("MASTER_SEED", MASTER_SEED),
    ("sway_mean", 1.05),
    ("sway_std", 0.05),
    ("sway_lower_bound", 1.0),
    ("note", "Aligned: RGEO=code5, pressure=max(0,..) w/ nominal refs, local RNG, export tidy series/band tables"),
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
# 9. 绘图（可选）
# =============================================================================
if PLOT:
    # --- Plot 1: Pareto w/ CI band
    plt.figure(figsize=(12, 7))
    plt.plot(df_pareto_series["x_time_years"], df_pareto_series["y_cost_trillion"],
             color=colors["hybrid"], linewidth=3, label="M3 Hybrid Mean")
    plt.fill_between(df_pareto_band["x_time_years"], df_pareto_band["y_lower_trillion"], df_pareto_band["y_upper_trillion"],
                     color=colors["hybrid"], alpha=0.2, label="M3 95% CI (Sway)")

    # M1 nominal point
    plt.scatter(t_m1_nom, cost_m1_nom, color=colors["elevator"], s=120, edgecolors='black', zorder=5, label="M1 Nominal")
    # M2 deterministic point
    plt.scatter(t_m2, cost_m2_tr, color=colors["rocket"], s=120, edgecolors='black', zorder=5, label="M2 Deterministic")

    plt.title(f"Pareto under Elevator Sway Uncertainty (Aligned, N={N_SAMPLES})", fontsize=14)
    plt.xlabel("Completion Time (Years)", fontsize=12)
    plt.ylabel("Total Project Cost (Trillion USD)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Trajectory w/ CI band
    fig, ax = plt.subplots(figsize=(12, 7))

    # M3 band + mean
    ax.fill_between(df_traj_band_m3["x_year"], df_traj_band_m3["y_lower_trillion"], df_traj_band_m3["y_upper_trillion"],
                    color=colors["hybrid"], alpha=0.2)
    ax.plot(df_traj_series_m3["x_year"], df_traj_series_m3["y_cost_trillion"], color=colors["hybrid"], linewidth=2.5, label="M3 Mean")

    # M1 band + mean
    ax.fill_between(df_traj_band_m1["x_year"], df_traj_band_m1["y_lower_trillion"], df_traj_band_m1["y_upper_trillion"],
                    color=colors["elevator"], alpha=0.2)
    ax.plot(df_traj_series_m1["x_year"], df_traj_series_m1["y_cost_trillion"], color=colors["elevator"], linewidth=2.5, label="M1 Mean")

    # M2 deterministic
    ax.plot(ys_m2, cs_m2, color=colors["rocket"], linewidth=2.5, label="M2 Deterministic")

    ax.set_title("Cost Trajectories under Uncertainty (Aligned)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Accumulated Cost (Trillion USD)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(START_YEAR, START_YEAR + 450)
    ax.legend(loc="upper left", frameon=False)

    # inset
    axins = inset_axes(ax, width="40%", height="40%", loc=1, borderpad=3)
    axins.fill_between(df_traj_band_m3["x_year"], df_traj_band_m3["y_lower_trillion"], df_traj_band_m3["y_upper_trillion"],
                       color=colors["hybrid"], alpha=0.2)
    axins.plot(df_traj_series_m3["x_year"], df_traj_series_m3["y_cost_trillion"], color=colors["hybrid"])
    axins.plot(ys_m2, cs_m2, color=colors["rocket"])

    axins.set_xlim(2050, 2220)
    axins.set_ylim(0, max(cost_m2_tr, np.nanmax(m3_hi)) * 1.05)
    axins.grid(True, linestyle=":", alpha=0.4)
    axins.set_title("Zoom: 2050-2220", fontsize=9)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--")

    plt.tight_layout()
    plt.show()

print("--- Code2 aligned run complete ---")