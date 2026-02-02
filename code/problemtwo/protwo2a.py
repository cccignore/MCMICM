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
# 2. 基地异质性配置 (基于真实建成时间的故障模型)
# =============================================================================
# 定义梯队参数: 设施越老，MTBF越短(易坏)，MTTR越长(难修)
TIER_PARAMS = {
    "Modern":   {"mtbf": 300, "mttr": 2},  
    "Standard": {"mtbf": 200, "mttr": 4}, 
    "Legacy":   {"mtbf": 140,  "mttr": 7}  
}

# 真实映射表
BASE_TIERS = {
    "Starbase (USA)": "Modern", "Mahia (NZL)": "Modern", "Alaska (USA)": "Modern",
    "Kourou (GUF)": "Standard", "Taiyuan (CHN)": "Standard", "Sriharikota (IND)": "Standard",
    "Vandenberg (USA)": "Legacy", "Wallops (USA)": "Legacy", 
    "Cape Canaveral (USA)": "Legacy", "Baikonur (KAZ)": "Legacy"
}

def get_rocket_stats(dv, isp, alpha, is_elevator=False):
    r = np.exp((dv * 1000) / (isp * G0))
    kappa = r * (1 + alpha)
    cost = alpha * C_DRY + (r - 1) * (1 + alpha) * C_PROP
    if is_elevator:
        delta_epsilon = MU * (1/RE - 1/RGEO) * 1e6
        kwh_per_ton = (1000 * delta_epsilon) / (ETA_E * 3.6e6)
        cost += kappa * (kwh_per_ton * PI_E)
    return kappa, cost

# 初始化数据
k_g, cost_g = get_rocket_stats(DV_G, ISP_G, ALPHA_G, True)
yearly_payload_g = (N_PORTS * U_PORT) / k_g # 电梯是理想的，不打折

base_configs_list = []
for name, lat in BASES_DATA.items():
    v_rot = V_ROT_EQUATOR * np.cos(np.radians(lat))
    dv_eff = DV_BASE - v_rot
    k, c = get_rocket_stats(dv_eff, ISP_E, ALPHA_E, False)
    
    # 理论年运力 (Ideal Capacity)
    max_y_ideal = (LAMBDA_J * 365 * L_MAX) / k
    
    # 获取可靠性参数
    tier = BASE_TIERS.get(name, "Standard")
    params = TIER_PARAMS[tier]
    
    # 计算可用度 Availability = MTBF / (MTBF + MTTR)
    denom = params['mtbf'] + params['mttr']
    if denom == 0: availability = 1.0
    else: availability = params['mtbf'] / denom
    
    # 期望有效运力 (Effective Capacity) - 用于宏观规划
    # 这就是"系统感知到火箭不可靠"的数学体现
    # max_y_effective = max_y_ideal * availability
    max_y_effective = max_y_ideal  # 保留真实故障波动：规划层不预扣维修时间（避免双重惩罚）
    
    base_configs_list.append({
        "Name": name, "Cost": c, 
        "Yearly_Ideal": max_y_ideal, 
        "Yearly_Effective": max_y_effective, # 规划用这个
        "MTBF": params['mtbf'], "MTTR": params['mttr'],
        "Tier": tier
    })

df_bases = pd.DataFrame(base_configs_list).sort_values("Cost")
# total_yearly_rockets_effective = df_bases["Yearly_Effective"].sum()
total_yearly_rockets_effective = df_bases["Yearly_Ideal"].sum()

# 重新计算工期极限 (基于有效运力)
T_MIN = M_TOTAL / (yearly_payload_g + total_yearly_rockets_effective)
T_MAX = M_TOTAL / yearly_payload_g

# =============================================================================
# 3. 核心算法: 考虑故障风险的宏观规划求解器
# =============================================================================
def solve_allocation_with_reliability(time_target, total_mass):
    """
    使用'期望有效运力'进行分配决策。
    因为火箭不可靠，Effective Capacity 变小了，算法会自动减少对火箭的依赖。
    """
    final_marginal_discount = 0.0 
    
    for i in range(10):
        # 积分平均折扣
        avg_discount = final_marginal_discount * 0.5
        effective_base_costs = df_bases["Cost"] * (1 - avg_discount)
        avg_rocket_cost = effective_base_costs.mean()
        
        rem = total_mass
        cost_iter = 0
        mass_ele = 0
        mass_roc = 0
        
        if avg_rocket_cost < cost_g:
            # 极端情况优先火箭
            can_take_roc = min(rem, total_yearly_rockets_effective * time_target)
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
            for idx, b in df_bases.iterrows():
                if needed_rocket <= 0: break
                # 注意：这里分配的是 Effective Capacity，意味着算法已经预留了维修时间
                # can_take = min(needed_rocket, b["Yearly_Effective"] * time_target)
                can_take = min(needed_rocket, b["Yearly_Ideal"] * time_target)  # 保留真实故障：规划不预扣维修
                cost_iter += can_take * b["Cost"] * (1 - avg_discount)
                mass_roc += can_take
                needed_rocket -= can_take
        
        new_marginal_discount = SCALE_DISCOUNT * (mass_roc / total_mass)
        if abs(new_marginal_discount - final_marginal_discount) < 0.0001:
            return cost_iter, mass_roc, mass_ele, new_marginal_discount
        final_marginal_discount = new_marginal_discount
        
    return cost_iter, mass_roc, mass_ele, final_marginal_discount

# =============================================================================
# 4. 离散事件仿真器 (Stochastic Discrete Event Simulator)
# =============================================================================
def simulate_rocket_trajectory(target_mass, seed=42, pace_years=None, disable_failures=False):
    """
    火箭故障仿真引擎
    修正点：取消可用度补偿逻辑，火箭仅按理想物理极限或目标配速执行，不考虑维修时间损失的预补偿。
    新增：disable_failures=True 时，跳过故障逻辑，作为“无故障基线”。
    """
    if target_mass <= 0:
        return np.array([0]), np.array([0]), np.array([0])
        # --- 进度打印（如果 progress_tick 存在） ---
    try:
        progress_tick(f"seed={seed}, mass={target_mass/1e6:.1f}Mt, pace={pace_years}, nofail={disable_failures}")
    except NameError:
        pass
    # ----------------------------------------    
    np.random.seed(seed)
    
    num_bases = len(df_bases)
    bases_status = np.ones(num_bases, dtype=int)
    repair_counters = np.zeros(num_bases, dtype=int)
    
    daily_caps = df_bases["Yearly_Ideal"].values / 365.0 
    
    unit_costs = df_bases["Cost"].values
    mtbfs = df_bases["MTBF"].values
    mttrs = df_bases["MTTR"].values

    # --- 修改部分：取消可用度补偿 ---
    if pace_years is not None:
        # 1. 计算理论上每天需要运多少 (仅基于目标工期)
        daily_needed_raw = target_mass / (pace_years * 365.0)
        
        # 2. 将补偿系数强制设为 1.0 (不再为了对冲故障而提前“加班”)
        avg_availability = 1.0 

        # 3. 修正后的每日目标 (此时 daily_needed_corrected == daily_needed_raw)
        daily_needed_corrected = daily_needed_raw / avg_availability
        
        daily_max_possible = np.sum(daily_caps)
        
        # 如果需求在物理极限内，则降速；否则全速。
        if daily_max_possible > daily_needed_corrected:
            throttle_factor = daily_needed_corrected / daily_max_possible
            daily_caps = daily_caps * throttle_factor
    # ----------------------------
    
    days = [0]
    costs = [0]
    masses = [0]
    
    current_mass = 0.0
    total_acc_cost = 0.0  
    
    day_count = 0
    
    while current_mass < target_mass:
        day_count += 1
        daily_mass_this_day = 0.0
        
        for i in range(num_bases):
            # 1. 维修逻辑
            if bases_status[i] == 0:
                repair_counters[i] -= 1
                if repair_counters[i] <= 0:
                    bases_status[i] = 1
                else:
                    continue 
            
            # 2. 故障逻辑判定
            if not disable_failures:
                if mttrs[i] > 0:
                    if np.random.rand() < (1.0 / mtbfs[i]):
                        bases_status[i] = 0
                        r_days = int(np.random.normal(mttrs[i], mttrs[i]*0.2))
                        repair_counters[i] = max(1, r_days)
                        continue 
            
            # 3. 正常发射
            prod = daily_caps[i]
            if current_mass + daily_mass_this_day + prod > target_mass:
                prod = target_mass - (current_mass + daily_mass_this_day)
            
            midpoint_mass = current_mass + daily_mass_this_day + (prod / 2.0)
            current_instant_discount = SCALE_DISCOUNT * (midpoint_mass / M_TOTAL)
            total_acc_cost += prod * unit_costs[i] * (1 - current_instant_discount)
            
            daily_mass_this_day += prod
            
        current_mass += daily_mass_this_day
        
        if day_count % 30 == 0 or current_mass >= target_mass:
            days.append(day_count / 365.0)
            masses.append(current_mass)
            costs.append(total_acc_cost / 1e12)
            
        if day_count > 1000 * 365: break 
        
    return np.array(days), np.array(costs), np.array(masses)

# =============================================================================
# 4.5 Monte Carlo 工具 (固定计划，不做补偿，不做保守优化，只评估意外波动)
# =============================================================================
def apply_pressure_penalty(raw_cost_usd, t_real):
    pressure = (T_MAX - t_real) / (T_MAX - T_MIN)
    return raw_cost_usd * (1 + 0.5 * (pressure ** 2))

def eval_plan_monte_carlo(mass_roc, mass_ele, pace_years, seeds, disable_failures=False):
    """
    固定计划(mass_roc, mass_ele)不变，跑多次随机仿真评估:
    - 实际完工时间分布 (总项目 = max(火箭完工, 电梯完工))
    - 总成本分布 (火箭仿真成本 + 电梯线性成本 + 压力罚款)
    返回: list of (t_total_real, total_cost_TUSD)
    """
    ele_rate = yearly_payload_g
    t_ele = 0.0
    if ele_rate > 0:
        t_ele = mass_ele / ele_rate

    results = []

    for sd in seeds:
        roc_days, roc_costs, _ = simulate_rocket_trajectory(
            mass_roc, seed=int(sd), pace_years=pace_years, disable_failures=disable_failures
        )
        t_roc = roc_days[-1]
        t_total = max(t_roc, t_ele)

        # 火箭成本 (TUSD)
        cost_roc = roc_costs[-1]  # 已经是 /1e12

        # 电梯成本 (TUSD)
        cost_ele = (mass_ele * cost_g) / 1e12

        raw_cost_total = (cost_roc + cost_ele) * 1e12

        # 压力罚款按“实际完工时间”罚，体现意外拖延导致的惩罚
        final_cost_total = apply_pressure_penalty(raw_cost_total, t_total)

        results.append((t_total, final_cost_total / 1e12))

    return results

def build_fan_chart_rocket_only(target_mass, seeds, pace_years=None, t_grid=None, disable_failures=False):
    """
    火箭-only 的 fan chart:
    - 每次仿真得到一条(时间,累计成本)曲线
    - 每条曲线先按各自“最终压力罚款终点”做比例缩放(保持你原先的处理口径)
    - 再插值到公共 t_grid 上，取 P10/P50/P90
    同时返回若干抽样曲线(原始台阶效果)用于叠加展示。
    """
    curves = []
    t_ends = []
    final_costs = []
    raw_costs = []

    for sd in seeds:
        d, c, _ = simulate_rocket_trajectory(target_mass, seed=int(sd), pace_years=pace_years, disable_failures=disable_failures)
        t_real = d[-1]
        raw_cost = c[-1] * 1e12
        final_cost = apply_pressure_penalty(raw_cost, t_real)

        # 按终点含罚款比例缩放整条曲线（延续你原来 M2 的口径）
        if raw_cost > 0:
            scale = final_cost / raw_cost
        else:
            scale = 1.0
        c_scaled = c * scale  # 仍然是 TUSD

        curves.append((d, c_scaled))
        t_ends.append(t_real)
        final_costs.append(final_cost / 1e12)
        raw_costs.append(raw_cost / 1e12)

    if t_grid is None:
        max_t = float(np.max(t_ends))
        t_grid = np.linspace(0, max_t, 600)

    mat = []
    for (d, c) in curves:
        mat.append(np.interp(t_grid, d, c))
    mat = np.array(mat)

    p10 = np.percentile(mat, 10, axis=0)
    p50 = np.percentile(mat, 50, axis=0)
    p90 = np.percentile(mat, 90, axis=0)

    return t_grid, p10, p50, p90, curves, np.array(t_ends), np.array(final_costs), np.array(raw_costs)

def build_fan_chart_hybrid(mass_roc, mass_ele, pace_years, seeds, t_grid=None, disable_failures=False):
    """
    混合策略 M3 的 fan chart:
    - 火箭部分随机仿真得到 roc 曲线
    - 电梯部分线性 determinisitic
    - 合并后按“总项目实际完工时间”计算压力罚款，并按终点比例缩放整条曲线（与原先M3口径一致）
    """
    ele_rate = yearly_payload_g
    t_ele = 0.0
    if ele_rate > 0:
        t_ele = mass_ele / ele_rate

    ele_cost_total = (mass_ele * cost_g) / 1e12  # TUSD

    curves_total = []
    t_total_ends = []
    final_costs = []
    raw_costs = []

    roc_curves = []

    for sd in seeds:
        roc_days, roc_costs, _ = simulate_rocket_trajectory(mass_roc, seed=int(sd), pace_years=pace_years, disable_failures=disable_failures)

        # 电梯累计成本曲线（按时间计算，t_ele 后不再增加）
        # 这里不急着插值，后面统一插值到 t_grid
        roc_curves.append((roc_days, roc_costs))

        t_roc = roc_days[-1]
        t_total = max(t_roc, t_ele)

        # 合并终点成本（raw，不含压力罚款）
        cost_roc_end = roc_costs[-1]  # TUSD
        cost_total_end = cost_roc_end + ele_cost_total
        raw_cost_total = cost_total_end * 1e12

        final_cost_total = apply_pressure_penalty(raw_cost_total, t_total)

        # 终点缩放
        if raw_cost_total > 0:
            scale = final_cost_total / raw_cost_total
        else:
            scale = 1.0

        # 先在各自时间轴上生成“总成本曲线”，再缩放
        # （后面统一插值到 t_grid）
        # 电梯部分：t <= t_ele 线性增长，>t_ele 维持常数
        # 火箭部分：直接用仿真曲线
        # 合并：火箭插值到相同时间点 + 电梯解析
        curves_total.append((roc_days, roc_costs, scale, t_ele, ele_cost_total))

        t_total_ends.append(t_total)
        final_costs.append(final_cost_total / 1e12)
        raw_costs.append(raw_cost_total / 1e12)

    if t_grid is None:
        max_t = float(np.max(t_total_ends))
        t_grid = np.linspace(0, max_t, 600)

    mat = []
    sample_total_curves = []
    for (roc_days, roc_costs, scale, t_ele, ele_cost_total) in curves_total:
        roc_interp = np.interp(t_grid, roc_days, roc_costs)
        # 电梯曲线：线性到 t_ele
        if t_ele > 0:
            ele_curve = np.minimum(t_grid * (ele_cost_total / t_ele), ele_cost_total)
        else:
            ele_curve = np.zeros_like(t_grid)
        total_curve = (roc_interp + ele_curve) * scale
        mat.append(total_curve)

        # 同时保留一条“抽样用的台阶总曲线”（用原 roc_days 作为时间基准）
        # 这里为了显示台阶，我们在 roc_days 上计算对应总成本（电梯解析 + roc_costs）
        if t_ele > 0:
            ele_on_roc_days = np.minimum(roc_days * (ele_cost_total / t_ele), ele_cost_total)
        else:
            ele_on_roc_days = np.zeros_like(roc_days)
        total_on_roc_days = (roc_costs + ele_on_roc_days) * scale
        sample_total_curves.append((roc_days, total_on_roc_days))

    mat = np.array(mat)

    p10 = np.percentile(mat, 10, axis=0)
    p50 = np.percentile(mat, 50, axis=0)
    p90 = np.percentile(mat, 90, axis=0)

    return t_grid, p10, p50, p90, sample_total_curves, np.array(t_total_ends), np.array(final_costs), np.array(raw_costs)

# =============================================================================
# 5. 数据生成
# =============================================================================
# 5.0 进度显示 (Progress)

print("正在执行异质性故障仿真...")

# Monte Carlo 参数
N_MC_PARETO = 30     # 帕累托带（每个T跑多少次，太大很慢）
N_MC_TRAJ = 50      # 轨迹带/相对变化（建议 200）
N_SAMPLE_LINES = 10  # 抽样叠加展示的“台阶线”数量

SEEDS_PARETO = np.arange(1000, 1000 + N_MC_PARETO)
SEEDS_TRAJ = np.arange(2000, 2000 + N_MC_TRAJ)

# --- A. 帕累托前沿 (规划线 + 无故障基线 + 故障带) ---
t_axis = np.linspace(T_MIN, T_MAX * 1.1, 20) # 稍微延长以展示延误
pareto_data = []
# =============================================================================
SIM_TOTAL = (
    len(t_axis) * (1 + len(SEEDS_PARETO))  # 帕累托: 每个T 1次无故障 + N次有故障
    + 1                                    # M2 单次示例
    + len(SEEDS_TRAJ)                      # M2 fan chart
    + 1                                    # M2 无故障基线
    + 1                                    # M3 单次示例(roc)
    + len(SEEDS_TRAJ)                      # M3 fan chart
    + 1                                    # M3 无故障基线
)

SIM_DONE = 0

def progress_tick(msg=""):
    global SIM_DONE
    SIM_DONE += 1
    # 控制打印频率：每 25 次打印一次 + 最后一次必打
    if SIM_DONE == 1 or SIM_DONE % 25 == 0 or SIM_DONE == SIM_TOTAL:
        print(f"[Sim {SIM_DONE}/{SIM_TOTAL}] {msg}")
# =============================================================================
for T in t_axis:
    # 这里的 solve 已经使用了 Effective Capacity，所以结果已经包含了"可靠性惩罚"
    raw_cost, m_roc, m_ele, avg_discount = solve_allocation_with_reliability(T, M_TOTAL)
    
    # 加上工期压力惩罚
    pressure = (T_MAX - T) / (T_MAX - T_MIN)
    final_cost = raw_cost * (1 + 0.5 * (pressure ** 2))
    
    pareto_data.append({
        "Time": T, 
        "Cost": final_cost / 1e12
    })

df_pareto = pd.DataFrame(pareto_data)

# 新增：无故障基线曲线（disable_failures=True）
pareto_nofail = []
for T in t_axis:
    _, m_roc, m_ele, _ = solve_allocation_with_reliability(T, M_TOTAL)

    sims0 = eval_plan_monte_carlo(
        mass_roc=m_roc, mass_ele=m_ele, pace_years=T,
        seeds=[0], disable_failures=True
    )
    cost0 = sims0[0][1]

    pareto_nofail.append({
        "Time": T,
        "Cost": cost0
    })

df_pareto_nofail = pd.DataFrame(pareto_nofail)

# 新增：故障带（P10~P90）+ 中位数（P50）
pareto_band = []
for T in t_axis:
    _, m_roc, m_ele, _ = solve_allocation_with_reliability(T, M_TOTAL)

    sims = eval_plan_monte_carlo(
        mass_roc=m_roc, mass_ele=m_ele, pace_years=T,
        seeds=SEEDS_PARETO, disable_failures=False
    )
    costs = np.array([x[1] for x in sims])

    pareto_band.append({
        "Time": T,
        "Cost_P10": np.percentile(costs, 10),
        "Cost_P50": np.percentile(costs, 50),
        "Cost_P90": np.percentile(costs, 90),
    })

df_pareto_band = pd.DataFrame(pareto_band)

# --- B. 关键点与仿真 ---

# 1. Method 1 (Elevator Only) - 理想线性 (无摇晃)
t_m1 = T_MAX
cost_m1 = (M_TOTAL * cost_g) / 1e12
# 仿真数据生成 (线性)
years_m1 = np.linspace(0, t_m1, 200)
costs_m1 = years_m1 * (cost_m1 / t_m1)
years_m1 += START_YEAR

# 2. Method 2 (Rocket Only) - 随机故障仿真
# 目标: 运送 M_TOTAL
print("  Simulating Method 2 (Stochastic)...")

# 原单次仿真（保留）
m2_days, m2_costs, _ = simulate_rocket_trajectory(M_TOTAL, seed=101)

# 加上压力惩罚 (基于实际耗时)
t_m2_real = m2_days[-1]
raw_cost_m2 = m2_costs[-1] * 1e12
pressure_m2 = (T_MAX - t_m2_real) / (T_MAX - T_MIN)
final_cost_m2 = raw_cost_m2 * (1 + 0.5 * pressure_m2**2)
cost_m2 = final_cost_m2 / 1e12
# 调整仿真曲线的高度以匹配含罚款的终点
m2_costs = m2_costs * (final_cost_m2 / raw_cost_m2)
years_m2 = m2_days + START_YEAR

# 新增：M2 fan chart + 抽样线（基于多次仿真）
t_grid_m2, m2_p10, m2_p50, m2_p90, m2_curves_all, m2_t_ends, m2_final_costs, _ = build_fan_chart_rocket_only(
    M_TOTAL, seeds=SEEDS_TRAJ, pace_years=None, t_grid=None, disable_failures=False
)
years_grid_m2 = t_grid_m2 + START_YEAR

# 无故障 M2 基线（用于相对变化图）
t_grid_m2_nf, m2_nf_p10, m2_nf_p50, m2_nf_p90, m2_nf_curves_all, m2_nf_t_ends, m2_nf_final_costs, _ = build_fan_chart_rocket_only(
    M_TOTAL, seeds=[0], pace_years=None, t_grid=t_grid_m2, disable_failures=True
)
m2_t0 = float(m2_nf_t_ends[0])
m2_c0 = float(m2_nf_final_costs[0])

# 3. Method 3 (Hybrid) - 混合仿真
# 目标: 150年 (期望)
t_target = 150.0
# 规划分配 (使用有效运力规划)
_, m_roc_plan, m_ele_plan, _ = solve_allocation_with_reliability(t_target, M_TOTAL)
print(f"  Simulating Method 3 (Hybrid): Rockets={m_roc_plan/1e6:.1f}Mt, Elevator={m_ele_plan/1e6:.1f}Mt")

# 3.1 电梯部分 (线性)
# 电梯速率
ele_rate = yearly_payload_g

# 3.2 火箭部分 (随机，强制降速)
m3_roc_days, m3_roc_costs, m3_roc_masses = simulate_rocket_trajectory(m_roc_plan, seed=101, pace_years=t_target)

# 3.3 合并曲线
# 我们需要一个统一的时间轴来叠加成本
max_time = max(m_ele_plan / ele_rate, m3_roc_days[-1])
common_time = np.linspace(0, max_time, 500)

# 插值火箭成本
interp_roc_cost = np.interp(common_time, m3_roc_days, m3_roc_costs)
# 计算电梯成本
ele_cost_total = (m_ele_plan * cost_g) / 1e12
interp_ele_cost = np.minimum(common_time * (ele_cost_total / (m_ele_plan/ele_rate)), ele_cost_total)

total_m3_cost = interp_roc_cost + interp_ele_cost
years_m3 = common_time + START_YEAR
t_m3_real = common_time[-1]
cost_m3 = total_m3_cost[-1]

# 压力惩罚
raw_cost_m3 = cost_m3 * 1e12
pressure_m3 = (T_MAX - t_m3_real) / (T_MAX - T_MIN)
final_cost_m3 = raw_cost_m3 * (1 + 0.5 * pressure_m3**2)
cost_m3_final = final_cost_m3 / 1e12
# 调整曲线
costs_m3 = total_m3_cost * (cost_m3_final / cost_m3)

# 新增：M3 fan chart + 抽样线（基于多次仿真，计划不变，只看意外波动）
t_grid_m3, m3_p10, m3_p50, m3_p90, m3_sample_total_curves, m3_t_ends, m3_final_costs, _ = build_fan_chart_hybrid(
    mass_roc=m_roc_plan, mass_ele=m_ele_plan, pace_years=t_target, seeds=SEEDS_TRAJ, t_grid=None, disable_failures=False
)
years_grid_m3 = t_grid_m3 + START_YEAR

# 无故障 M3 基线（用于相对变化图）
t_grid_m3_nf, m3_nf_p10, m3_nf_p50, m3_nf_p90, m3_nf_sample_total_curves, m3_nf_t_ends, m3_nf_final_costs, _ = build_fan_chart_hybrid(
    mass_roc=m_roc_plan, mass_ele=m_ele_plan, pace_years=t_target, seeds=[0], t_grid=t_grid_m3, disable_failures=True
)
m3_t0 = float(m3_nf_t_ends[0])
m3_c0 = float(m3_nf_final_costs[0])

# =============================================================================
# 6. 绘图 1: 帕累托前沿 (无故障基线 + 故障带 + 中位数)
# =============================================================================
plt.figure(figsize=(12, 7))

# 原规划线（保留）
# plt.plot(df_pareto["Time"], df_pareto["Cost"], 'b-', linewidth=3, label='Method 3: Hybrid (Reliability Adjusted)')
# plt.fill_between(df_pareto["Time"], df_pareto["Cost"], color='blue', alpha=0.1)

# 无故障基线
plt.plot(df_pareto_nofail["Time"], df_pareto_nofail["Cost"], 'k--', linewidth=2.5, label='No-Failure Baseline (Simulation)')

# 故障带 + 中位数
plt.plot(df_pareto_band["Time"], df_pareto_band["Cost_P50"], 'b-', linewidth=3, label='Failure Case: Median (P50)')
plt.fill_between(df_pareto_band["Time"], df_pareto_band["Cost_P10"], df_pareto_band["Cost_P90"], color='blue', alpha=0.15, label='Failure Case: P10~P90 Band')

plt.scatter(t_m1, cost_m1, color='green', s=120, zorder=5, label='Method 1: Elevator Only')
plt.annotate(f'M1\n({t_m1:.1f}y, ${cost_m1:.1f}T)', (t_m1, cost_m1), 
             xytext=(-40, 10), textcoords='offset points', color='green', fontweight='bold')

plt.scatter(t_m2_real, cost_m2, color='red', s=120, zorder=5, label='Method 2: Rockets (One Realization)')
plt.annotate(f'M2\n({t_m2_real:.1f}y, ${cost_m2:.1f}T)', (t_m2_real, cost_m2), 
             xytext=(20, 10), textcoords='offset points', color='red', fontweight='bold')

# 标注火箭方案的延误
plt.arrow(t_m2_real - 20, cost_m2, 15, 0, head_width=2, head_length=3, fc='k', ec='k')
plt.text(t_m2_real - 25, cost_m2 + 5, "Delays due to\nLegacy Bases", ha='right', fontsize=9)

plt.title("Pareto Frontier: No-Failure Baseline vs Failure Uncertainty Band", fontsize=14)
plt.xlabel("Completion Time Target (Years)", fontsize=12)
plt.ylabel("Total Project Cost (Trillion USD)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig("pareto_frontier_uncertainty_band.png")
plt.show()

# =============================================================================
# 7. 绘图 2: 成本累积仿真 (fan chart + 抽样台阶线)
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

# M1
ax.plot(years_m1, costs_m1, color=colors['elevator'], linestyle='-', linewidth=2.5, label='Method 1: Elevator Only')
ax.scatter(years_m1[-1], costs_m1[-1], color=colors['elevator'], s=80, zorder=5, edgecolor='white', linewidth=1.5)
ax.annotate(f'{int(years_m1[-1])}\n${costs_m1[-1]:.1f}T', (years_m1[-1], costs_m1[-1]), 
             xytext=(-40, 10), textcoords="offset points", color=colors['elevator'], fontweight='bold', fontsize=9)

# M2 fan chart
ax.fill_between(years_grid_m2, m2_p10, m2_p90, alpha=0.18, label='Method 2: Rockets P10~P90 Band')
ax.plot(years_grid_m2, m2_p50, color=colors['rocket'], linestyle='-', linewidth=2.5, label='Method 2: Rockets Median (P50)')

# 抽样几条台阶线（展示随机“台阶效果”）
if len(m2_curves_all) > 0:
    sample_idx = np.linspace(0, len(m2_curves_all) - 1, min(N_SAMPLE_LINES, len(m2_curves_all))).astype(int)
    for si in sample_idx:
        d, c = m2_curves_all[si]
        ax.plot(d + START_YEAR, c, color=colors['rocket'], alpha=0.18, linewidth=1.2)

# 标注中位数终点
ax.scatter(years_grid_m2[-1], m2_p50[-1], color=colors['rocket'], s=80, zorder=5, edgecolor='white', linewidth=1.5)
ax.annotate(f'{int(years_grid_m2[-1])}\n${m2_p50[-1]:.1f}T', (years_grid_m2[-1], m2_p50[-1]), 
             xytext=(10, 5), textcoords="offset points", color=colors['rocket'], fontweight='bold', fontsize=9)

# M3 fan chart
ax.fill_between(years_grid_m3, m3_p10, m3_p90, alpha=0.18, label='Method 3: Hybrid P10~P90 Band')
ax.plot(years_grid_m3, m3_p50, color=colors['hybrid'], linestyle='-', linewidth=2.5, label='Method 3: Hybrid Median (P50)')

# 抽样几条台阶线（展示随机“台阶效果”）
if len(m3_sample_total_curves) > 0:
    sample_idx = np.linspace(0, len(m3_sample_total_curves) - 1, min(N_SAMPLE_LINES, len(m3_sample_total_curves))).astype(int)
    for si in sample_idx:
        d, c = m3_sample_total_curves[si]
        ax.plot(d + START_YEAR, c, color=colors['hybrid'], alpha=0.18, linewidth=1.2)

# 标注中位数终点
ax.scatter(years_grid_m3[-1], m3_p50[-1], color=colors['hybrid'], s=80, zorder=5, edgecolor='white', linewidth=1.5)
ax.annotate(f'{int(years_grid_m3[-1])}\n${m3_p50[-1]:.1f}T', (years_grid_m3[-1], m3_p50[-1]), 
             xytext=(10, 5), textcoords="offset points", color=colors['hybrid'], fontweight='bold', fontsize=9)

ax.set_title("Cumulative Cost Trajectories: Fan Charts + Sample Step Realizations", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Accumulated Cost (Trillion USD)", fontsize=12)
ax.grid(True, alpha=0.3)

# 仍保留原先的显示范围逻辑（尽量不改你原来的意图）
ax.set_xlim(START_YEAR, START_YEAR + 450)
ax.set_ylim(0, max(float(np.max(m2_p90)), float(np.max(m3_p90)), cost_m1)*1.1)
ax.legend(loc='upper left', frameon=False)
ax.axvspan(2050, 2200, color='gray', alpha=0.05)

# Inset Zoom
axins = inset_axes(ax, width="40%", height="40%", loc=1, borderpad=3)

axins.plot(years_m1, costs_m1, color=colors['elevator'], linestyle='-')
axins.plot(years_grid_m2, m2_p50, color=colors['rocket'], linestyle='-')
axins.plot(years_grid_m3, m3_p50, color=colors['hybrid'], linestyle='-')

axins.fill_between(years_grid_m2, m2_p10, m2_p90, alpha=0.15)
axins.fill_between(years_grid_m3, m3_p10, m3_p90, alpha=0.15)

# 抽样线（inset也画几条更直观）
if len(m2_curves_all) > 0:
    sample_idx = np.linspace(0, len(m2_curves_all) - 1, min(6, len(m2_curves_all))).astype(int)
    for si in sample_idx:
        d, c = m2_curves_all[si]
        axins.plot(d + START_YEAR, c, color=colors['rocket'], alpha=0.12, linewidth=1.0)

if len(m3_sample_total_curves) > 0:
    sample_idx = np.linspace(0, len(m3_sample_total_curves) - 1, min(6, len(m3_sample_total_curves))).astype(int)
    for si in sample_idx:
        d, c = m3_sample_total_curves[si]
        axins.plot(d + START_YEAR, c, color=colors['hybrid'], alpha=0.12, linewidth=1.0)

axins.set_xlim(2050, 2220)
axins.set_ylim(0, float(np.max(m2_p90[(years_grid_m2 >= 2050) & (years_grid_m2 <= 2220)]))*1.1 if np.any((years_grid_m2 >= 2050) & (years_grid_m2 <= 2220)) else float(m2_p90[-1])*1.1)
axins.grid(True, linestyle=':', alpha=0.4)
axins.set_title("Zoom: Uncertainty Band + Step Realizations", fontsize=9)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--")

plt.tight_layout()
plt.savefig("simulation_trajectories_fanchart.png", dpi=300)
plt.show()

# =============================================================================
# 8. 绘图 3: 相对无故障时的相对变化比例 (时间 + 成本)
# =============================================================================
# M2 相对变化
m2_dt = (m2_t_ends - m2_t0) / m2_t0
m2_dc = (m2_final_costs - m2_c0) / m2_c0

# M3 相对变化
m3_dt = (m3_t_ends - m3_t0) / m3_t0
m3_dc = (m3_final_costs - m3_c0) / m3_c0

fig3, (ax_t, ax_c) = plt.subplots(1, 2, figsize=(13, 5.8))

# 时间变化比例
ax_t.hist(m2_dt * 100, bins=30, alpha=0.55, label='Method 2: Rockets', color=colors['rocket'])
ax_t.hist(m3_dt * 100, bins=30, alpha=0.55, label='Method 3: Hybrid', color=colors['hybrid'])
ax_t.axvline(0, linestyle='--', linewidth=1.2, color='k', alpha=0.6)
ax_t.set_title("Relative Completion Time Change vs No-Failure Baseline", fontsize=12, fontweight='bold')
ax_t.set_xlabel("ΔTime (%)", fontsize=11)
ax_t.set_ylabel("Frequency", fontsize=11)
ax_t.grid(True, alpha=0.25)

# 成本变化比例
ax_c.hist(m2_dc * 100, bins=30, alpha=0.55, label='Method 2: Rockets', color=colors['rocket'])
ax_c.hist(m3_dc * 100, bins=30, alpha=0.55, label='Method 3: Hybrid', color=colors['hybrid'])
ax_c.axvline(0, linestyle='--', linewidth=1.2, color='k', alpha=0.6)
ax_c.set_title("Relative Cost Change vs No-Failure Baseline", fontsize=12, fontweight='bold')
ax_c.set_xlabel("ΔCost (%)", fontsize=11)
ax_c.set_ylabel("Frequency", fontsize=11)
ax_c.grid(True, alpha=0.25)

ax_c.legend(loc='upper right', frameon=False)

plt.tight_layout()
plt.savefig("relative_change_vs_nofailure.png", dpi=300)
plt.show()

print("--- 故障模型仿真完成 ---")
print(f"M2 单次示例 实际完工时间: {t_m2_real:.1f} 年 (受老旧基地拖累)")
print(f"M3 单次示例 实际完工时间: {t_m3_real:.1f} 年")

print("---- Monte Carlo 统计（对比无故障基线）----")
print(f"[M2] 无故障基线: T={m2_t0:.1f}y, Cost=${m2_c0:.2f}T")
print(f"[M2] P50: T={np.percentile(m2_t_ends,50):.1f}y, Cost=${np.percentile(m2_final_costs,50):.2f}T")
print(f"[M2] P90: T={np.percentile(m2_t_ends,90):.1f}y, Cost=${np.percentile(m2_final_costs,90):.2f}T")

print(f"[M3] 无故障基线: T={m3_t0:.1f}y, Cost=${m3_c0:.2f}T")
print(f"[M3] P50: T={np.percentile(m3_t_ends,50):.1f}y, Cost=${np.percentile(m3_final_costs,50):.2f}T")
print(f"[M3] P90: T={np.percentile(m3_t_ends,90):.1f}y, Cost=${np.percentile(m3_final_costs,90):.2f}T")