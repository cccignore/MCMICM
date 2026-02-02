import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# =============================================================================
# 0. 开关
# =============================================================================
PLOT = True
EXPORT_EXCEL = True
EXCEL_PATH = "plot_data_code5_elevator_breakdown_backfill.xlsx"

SCENARIO = "s5_elevator_breakdown_backfill"

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
SCALE_DISCOUNT = 0.2  # 折扣上限 20%
START_YEAR = 2050

# Scenario 5: Elevator breakdown params
MTBF_ELE = 300.0  # days
MTTR_ELE = 7.0    # days

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
# 3. 核心计算函数（与基底一致）
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
# 4. 规划层：乐观分配（不预扣电梯故障）
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
# 5. 电梯执行层：离散故障仿真（停机不计费，局部 RNG）
# =============================================================================
def simulate_elevator_with_breakdowns(
    target_mass,
    duration_years=None,
    unit_cost=None,
    seed=42,
    record_every_days=30
):
    """
    - 若 duration_years 给定：在该时间窗口内运行（硬截止），返回窗口末的 delivered mass/cost
    - 若 duration_years=None：一直跑到运完 target_mass（M1 用）
    - 停机不计费：repair days 不增加 cost
    """
    rng = np.random.default_rng(seed)

    daily_nominal = yearly_payload_g / 365.0
    unit_cost = cost_g if unit_cost is None else unit_cost

    if duration_years is None:
        max_days = int(5_000 * 365)  # 大上限防死循环
    else:
        max_days = int(np.ceil(duration_years * 365.0))

    delivered_mass = 0.0
    total_cost = 0.0

    days = [0.0]
    costs = [0.0]
    masses = [0.0]

    repair_days_remaining = 0

    for day in range(1, max_days + 1):
        if repair_days_remaining > 0:
            repair_days_remaining -= 1
        else:
            # 运行中：判定是否坏
            if rng.random() < (1.0 / MTBF_ELE):
                # 触发故障：维修期只耗时间，不计费
                r_days = int(rng.normal(MTTR_ELE, MTTR_ELE * 0.2))
                repair_days_remaining = max(1, r_days)
            else:
                # 正常运货
                prod = daily_nominal
                if target_mass is not None and (delivered_mass + prod > target_mass):
                    prod = target_mass - delivered_mass

                if prod > 0:
                    total_cost += prod * unit_cost
                    delivered_mass += prod

        # 记录
        if (day % record_every_days == 0) or (target_mass is not None and delivered_mass >= target_mass):
            days.append(day / 365.0)
            costs.append(total_cost / 1e12)
            masses.append(delivered_mass)

        # 跑到目标就停（仅在 duration_years=None 的 M1 场景需要）
        if duration_years is None and target_mass is not None and delivered_mass >= target_mass:
            break

    return np.array(days), np.array(costs), np.array(masses)

# =============================================================================
# 6. 火箭成本：按基地分段积分（保留基地单价差异 + 学习曲线连续积分）
# =============================================================================
def rocket_cost_piecewise_integral(total_rocket_mass, discount_base_mass=M_TOTAL):
    """
    假设火箭总运量 total_rocket_mass 在各基地按“从便宜到贵”吃满产能（Yearly）分配；
    折扣是全局的：discount(m) = S * (m / discount_base_mass), capped at 20%
    对每个基地 i：cost_i = ∫ P_i * (1 - discount(m)) dm  over its allocated mass segment
               = P_i * [Δm - S*(m2^2 - m1^2)/(2*B)]
    其中 m1/m2 为全局累计火箭运量区间端点。
    """
    if total_rocket_mass <= 0:
        return 0.0

    B = float(discount_base_mass)
    S = float(SCALE_DISCOUNT)

    # 这里不强制 clamp 在积分里逐点封顶，因为 max discount=20% 本来就是 S；
    # 如果 m > B，则 S*(m/B) > S，会超过上限，需要做分段：超过 B 后 discount 恒等 S。
    # 下面实现“封顶积分”。

    remaining = total_rocket_mass
    cumulative_m = 0.0
    total_cost = 0.0

    # helper: capped integral of (1 - min(S*m/B, S)) over [m1,m2]
    def integral_discount_term(m1, m2):
        # 返回 ∫ (1 - clamp(S*m/B, 0..S)) dm
        # clamp 后：在 m<=B 区间是 1 - S*m/B；在 m>=B 区间是 1 - S
        if m2 <= 0:
            return 0.0
        m1 = max(0.0, m1)
        m2 = max(0.0, m2)
        if m2 <= m1:
            return 0.0

        if B <= 0:
            # 退化：直接用最大折扣
            return (1.0 - S) * (m2 - m1)

        # 全在未封顶段
        if m2 <= B:
            return (m2 - m1) - S * (m2**2 - m1**2) / (2.0 * B)

        # 全在封顶段
        if m1 >= B:
            return (1.0 - S) * (m2 - m1)

        # 跨越 B：分两段
        part1 = (B - m1) - S * (B**2 - m1**2) / (2.0 * B)
        part2 = (1.0 - S) * (m2 - B)
        return part1 + part2

    # 分配到基地（便宜优先）
    for _, b in df_bases.iterrows():
        if remaining <= 0:
            break
        # 基地在 1 年里能运多少不重要，这里是“成本积分函数”
        # 产能约束会在外层（给定时间 T）检查
        take = min(remaining, remaining)  # 这里只分配总量，不做时间上限切割
        # 注意：这里的“分配”应由外层给定每基地分配量；
        # 但为了成本积分，我们通常在外层先做 capacity check，再用“全量按便宜优先”近似。
        # 这里将每基地的分配量按“便宜优先”吃满即可，需要由外层传入 per-base cap。
        # ——因此，本函数只用于“给定 per-base caps 的版本”，下面会有更严格版本。
        remaining -= take

    # 上面只是占位，不对：我们需要 per-base caps 版本，否则无法保留基地成本差异。
    raise RuntimeError("Use rocket_cost_given_caps_piecewise() instead (needs per-base caps).")

def rocket_cost_given_caps_piecewise(total_rocket_mass, caps_by_base, discount_base_mass=M_TOTAL):
    """
    caps_by_base：与 df_bases 同顺序的“该基地最多可承担的火箭运量”（由 Yearly*T 决定）
    按便宜优先分配到 cap，做 piecewise cost integral（含折扣封顶）。
    """
    if total_rocket_mass <= 0:
        return 0.0

    B = float(discount_base_mass)
    S = float(SCALE_DISCOUNT)

    def integral_discount_term(m1, m2):
        if m2 <= 0:
            return 0.0
        m1 = max(0.0, m1)
        m2 = max(0.0, m2)
        if m2 <= m1:
            return 0.0
        if B <= 0:
            return (1.0 - S) * (m2 - m1)
        if m2 <= B:
            return (m2 - m1) - S * (m2**2 - m1**2) / (2.0 * B)
        if m1 >= B:
            return (1.0 - S) * (m2 - m1)
        part1 = (B - m1) - S * (B**2 - m1**2) / (2.0 * B)
        part2 = (1.0 - S) * (m2 - B)
        return part1 + part2

    remaining = float(total_rocket_mass)
    cumulative_m = 0.0
    total_cost = 0.0

    for i, b in df_bases.iterrows():
        if remaining <= 0:
            break

        cap = float(caps_by_base[i])
        if cap <= 0:
            continue

        take = min(remaining, cap)
        m1 = cumulative_m
        m2 = cumulative_m + take

        # cost_i = P_i * ∫ (1 - discount(m)) dm
        p = float(b["Cost"])
        total_cost += p * integral_discount_term(m1, m2)

        cumulative_m = m2
        remaining -= take

    if remaining > 1e-6:
        # capacity不足
        return np.nan

    return total_cost

# =============================================================================
# 7. Pareto 数据：baseline planning + scenario5(backfill hard deadline)
# =============================================================================
print("Running scenario 5 (Elevator breakdown + Rocket backfill) aligned...")

t_axis = np.linspace(T_MIN, T_MAX, 100)

# A) baseline planning Pareto（不含故障）
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
        "note": "Planning optimistic (no breakdown)"
    })

# B) scenario5 Pareto：电梯故障 -> gap -> 火箭兜底（硬工期：按 T 交付，不允许提前完成）
S5_SEED = 2026
pareto_s5_rows = []

for T in t_axis:
    # planning optimistic
    _, m_roc_plan, m_ele_plan, _ = solve_allocation_with_integral_cost(T, M_TOTAL)

    # elevator executes for T years, tries to deliver planned elevator mass
    ele_days, ele_costs_tr, ele_masses = simulate_elevator_with_breakdowns(
        target_mass=m_ele_plan,
        duration_years=T,
        unit_cost=cost_g,
        seed=S5_SEED
    )
    m_ele_actual = float(ele_masses[-1])
    ele_cost_usd = float(ele_costs_tr[-1]) * 1e12

    gap = max(0.0, float(m_ele_plan - m_ele_actual))

    # rocket backfill total
    m_roc_needed = float(m_roc_plan + gap)

    # capacity check: rockets must deliver m_roc_needed within T
    caps = (df_bases["Yearly"].values * T)  # per-base cap
    if m_roc_needed > float(np.sum(caps)) + 1e-6:
        pareto_s5_rows.append({
            "scenario": SCENARIO,
            "curve": "scenario5_backfill_hard_deadline",
            "x_time_years": T,
            "y_cost_trillion": np.nan,
            "x_kind": "planned",
            "seed": S5_SEED,
            "note": "Infeasible: rocket capacity insufficient for backfill"
        })
        continue

    rocket_cost_usd = rocket_cost_given_caps_piecewise(
        total_rocket_mass=m_roc_needed,
        caps_by_base=caps,
        discount_base_mass=M_TOTAL
    )

    if np.isnan(rocket_cost_usd):
        pareto_s5_rows.append({
            "scenario": SCENARIO,
            "curve": "scenario5_backfill_hard_deadline",
            "x_time_years": T,
            "y_cost_trillion": np.nan,
            "x_kind": "planned",
            "seed": S5_SEED,
            "note": "Infeasible: rocket cap allocation failed"
        })
        continue

    raw_cost_usd = ele_cost_usd + rocket_cost_usd

    # hard deadline/no early finish => completion time is T
    final_cost_usd = raw_cost_usd * pressure_factor(T, T_MIN, T_MAX)

    pareto_s5_rows.append({
        "scenario": SCENARIO,
        "curve": "scenario5_backfill_hard_deadline",
        "x_time_years": T,
        "y_cost_trillion": final_cost_usd / 1e12,
        "x_kind": "planned",
        "seed": S5_SEED,
        "note": "Elevator breakdown + rocket backfill, hard deadline"
    })

df_pareto_series = pd.DataFrame(pareto_plan_rows + pareto_s5_rows)
df_pareto_band = pd.DataFrame(columns=[
    "scenario", "band", "x_time_years", "y_lower_trillion", "y_upper_trillion", "level", "stat_center"
])

# =============================================================================
# 8. 关键点与累计成本曲线（M1/M2/M3）
# =============================================================================
# ---- M1: elevator only w/ breakdowns (soft, runs until completion) ----
print("  Simulating M1 (Elevator only w/ breakdown) ...")
m1_days, m1_costs_tr, _ = simulate_elevator_with_breakdowns(
    target_mass=M_TOTAL, duration_years=None, unit_cost=cost_g, seed=2024
)
t_m1_real = float(m1_days[-1])
cost_m1_tr = float(m1_costs_tr[-1])

years_m1 = START_YEAR + m1_days
traj_m1 = m1_costs_tr  # already trillion

# ---- M2: rocket only deterministic (no breakdown). finish at t_m2 and cost via piecewise integral. ----
t_m2 = M_TOTAL / total_yearly_rockets
caps_m2 = df_bases["Yearly"].values * t_m2
rocket_cost_m2_usd = rocket_cost_given_caps_piecewise(M_TOTAL, caps_m2, discount_base_mass=M_TOTAL)

pressure_m2 = pressure_factor(t_m2, T_MIN, T_MAX)
final_cost_m2_usd = rocket_cost_m2_usd * pressure_m2
cost_m2_tr = final_cost_m2_usd / 1e12

# build a smooth trajectory for M2 using the same integral (uniform time fraction, mass fraction)
years_m2 = np.linspace(0.0, t_m2, 300)
traj_m2 = []
for t in years_m2:
    m = M_TOTAL * (t / t_m2) if t_m2 > 0 else 0.0
    caps_t = df_bases["Yearly"].values * t  # by time t
    c_usd = rocket_cost_given_caps_piecewise(m, caps_t, discount_base_mass=M_TOTAL)
    traj_m2.append((c_usd * pressure_m2) / 1e12)  # scale by same pressure factor
years_m2 = START_YEAR + years_m2
traj_m2 = np.array(traj_m2)

# ---- M3: hybrid plan 150y + elevator breakdown + backfill rockets, hard deadline (no early finish) ----
print("  Simulating M3 (Hybrid backfill, hard deadline) ...")
t_plan_m3 = 150.0
_, m_roc_plan, m_ele_plan, _ = solve_allocation_with_integral_cost(t_plan_m3, M_TOTAL)

# elevator runs for 150 years
m3_ele_days, m3_ele_costs_tr, m3_ele_masses = simulate_elevator_with_breakdowns(
    target_mass=m_ele_plan,
    duration_years=t_plan_m3,
    unit_cost=cost_g,
    seed=2025
)
m_ele_actual = float(m3_ele_masses[-1])
gap = max(0.0, float(m_ele_plan - m_ele_actual))
m_roc_needed = float(m_roc_plan + gap)

# capacity check for 150 years
caps_m3 = df_bases["Yearly"].values * t_plan_m3
if m_roc_needed > float(np.sum(caps_m3)) + 1e-6:
    # infeasible: still output something but mark as NaN for M3
    t_m3_used = t_plan_m3
    cost_m3_tr = np.nan
    years_m3 = START_YEAR + np.array([0.0, t_plan_m3])
    traj_m3 = np.array([0.0, np.nan])
else:
    # build rocket cost curve assuming uniform delivery over the 150y window
    common_t = m3_ele_days.copy()  # use elevator recorded grid for alignment
    common_t[common_t > t_plan_m3] = t_plan_m3

    rocket_cost_curve_tr = []
    for t in common_t:
        m_delivered = m_roc_needed * (t / t_plan_m3) if t_plan_m3 > 0 else 0.0
        caps_t = df_bases["Yearly"].values * t
        c_usd = rocket_cost_given_caps_piecewise(m_delivered, caps_t, discount_base_mass=M_TOTAL)
        rocket_cost_curve_tr.append(c_usd / 1e12)
    rocket_cost_curve_tr = np.array(rocket_cost_curve_tr)

    # total raw curve (trillion)
    total_raw_curve_tr = m3_ele_costs_tr + rocket_cost_curve_tr

    # hard deadline: completion time is exactly 150y
    pressure_m3 = pressure_factor(t_plan_m3, T_MIN, T_MAX)

    raw_total_usd = float(total_raw_curve_tr[-1]) * 1e12
    final_total_usd = raw_total_usd * pressure_m3
    cost_m3_tr = final_total_usd / 1e12

    # scale curve to end at final (pressure-included) cost
    scale = (final_total_usd / raw_total_usd) if raw_total_usd > 0 else 1.0
    traj_m3 = total_raw_curve_tr * scale
    years_m3 = START_YEAR + common_t

# points
df_pareto_points = pd.DataFrame([
    {"scenario": SCENARIO, "method": "M1", "time_years": t_m1_real, "cost_trillion": cost_m1_tr, "time_kind": "real", "note": "Elevator only w/ breakdowns (no backfill)"},
    {"scenario": SCENARIO, "method": "M2", "time_years": t_m2, "cost_trillion": cost_m2_tr, "time_kind": "planned", "note": "Rocket only deterministic (piecewise integral)"},
    {"scenario": SCENARIO, "method": "M3", "time_years": t_plan_m3, "cost_trillion": cost_m3_tr, "time_kind": "planned", "note": "Hybrid: elevator breakdown + rocket backfill (hard deadline)"},
])

# traj series
df_traj_series = pd.concat([
    pd.DataFrame({"scenario": SCENARIO, "curve": "M1", "x_year": years_m1, "y_cost_trillion": traj_m1, "seed": 2024, "note": "Elevator breakdown stochastic"}),
    pd.DataFrame({"scenario": SCENARIO, "curve": "M2", "x_year": years_m2, "y_cost_trillion": traj_m2, "seed": np.nan, "note": "Rocket deterministic piecewise integral"}),
    pd.DataFrame({"scenario": SCENARIO, "curve": "M3", "x_year": years_m3, "y_cost_trillion": traj_m3, "seed": 2025, "note": "Hybrid backfill hard deadline"}),
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
    ("MTBF_ELE_days", MTBF_ELE),
    ("MTTR_ELE_days", MTTR_ELE),
    ("note", "Scenario5: Planning optimistic; elevator breakdown stochastic (downtime no charge); gap backfilled by rockets within same planned T (hard deadline, no early finish). Rocket cost uses per-base piecewise integral with discount capped at 20%. Pressure=max(0,..). Local RNG."),
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
    df_s5 = df_pareto_series[df_pareto_series["curve"] == "scenario5_backfill_hard_deadline"]

    plt.plot(df_plan["x_time_years"], df_plan["y_cost_trillion"], color=colors["hybrid"], linewidth=3,
             label="Hybrid Pareto (Planning, no breakdown)")
    plt.fill_between(df_plan["x_time_years"], df_plan["y_cost_trillion"], color=colors["hybrid"], alpha=0.10)

    plt.plot(df_s5["x_time_years"], df_s5["y_cost_trillion"], color="magenta", linewidth=3,
             label="Scenario 5 Pareto (Breakdown + Backfill, hard deadline)")
    plt.fill_between(df_s5["x_time_years"], df_s5["y_cost_trillion"], color="magenta", alpha=0.10)

    for _, r in df_pareto_points.iterrows():
        c = {"M1": colors["elevator"], "M2": colors["rocket"], "M3": colors["hybrid"]}[r["method"]]
        plt.scatter(r["time_years"], r["cost_trillion"], color=c, s=120, zorder=5, edgecolors='black')
        plt.annotate(f'{r["method"]}\n({r["time_years"]:.1f}y, ${r["cost_trillion"]:.1f}T)',
                     (r["time_years"], r["cost_trillion"]),
                     xytext=(10, 10), textcoords='offset points',
                     color=c, fontweight='bold', fontsize=9)

    plt.title("Pareto Frontier: Elevator Breakdown + Rocket Backfill (Aligned)", fontsize=14)
    plt.xlabel("Completion Time (Years)")
    plt.ylabel("Total Cost (Trillion USD)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    # Trajectory
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(years_m1, traj_m1, color=colors["elevator"], linewidth=2.5, label="M1: Elevator (Breakdown)")
    ax.plot(years_m2, traj_m2, color=colors["rocket"], linewidth=2.5, label="M2: Rockets (Deterministic)")
    ax.plot(years_m3, traj_m3, color=colors["hybrid"], linewidth=2.5, label="M3: Hybrid Backfill (Hard deadline)")

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
    axins.set_ylim(0, np.nanmax([np.nanmax(traj_m1), np.nanmax(traj_m2), np.nanmax(traj_m3)]) * 1.1)
    axins.grid(True, linestyle=":", alpha=0.4)
    axins.set_title("Zoom: 2050-2220", fontsize=9)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--")

    plt.tight_layout()
    plt.show()

print("--- Code5 aligned run complete ---")