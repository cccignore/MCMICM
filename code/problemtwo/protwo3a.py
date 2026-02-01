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

# [新增] 第三类模型：飞行失败参数
P_FAIL_CONST = 0.01          # 故障率: 1% (均值)
P_FAIL_STD = 0.003           # [新增] 故障率标准差: 0.3% (用于敏感性分析)
C_PAYLOAD_PER_TON = 1e5       # 货物价值: $100,000/ton (大宗精密建材)

BASES_DATA = {
    "Alaska (USA)": 57.43528, "Vandenberg (USA)": 34.75133,
    "Starbase (USA)": 25.99700, "Cape Canaveral (USA)": 28.48889,
    "Wallops (USA)": 37.84333, "Baikonur (KAZ)": 45.96500,
    "Kourou (GUF)": 5.16900, "Sriharikota (IND)": 13.72000,
    "Taiyuan (CHN)": 38.84910, "Mahia (NZL)": -39.26085
}

# 绘图配色
colors = {
    'rocket': '#D95F02',   
    'elevator': '#1B9E77', 
    'hybrid': '#7570B3'    
}

# =============================================================================
# 2. 核心计算函数
# =============================================================================
def get_rocket_stats(dv, isp, alpha, is_elevator=False):
    r = np.exp((dv * 1000) / (isp * G0))
    kappa = r * (1 + alpha)
    cost = alpha * C_DRY + (r - 1) * (1 + alpha) * C_PROP
    if is_elevator:
        delta_epsilon = MU * (1/RE - 1/RGEO) * 1e6
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
df_bases = pd.DataFrame(base_list).sort_values("Cost")
total_yearly_rockets = df_bases["Yearly"].sum()
T_MIN = M_TOTAL / (yearly_payload_g + total_yearly_rockets)
T_MAX = M_TOTAL / yearly_payload_g

# 核心算法: 动态迭代求解器 (积分修正版)
def solve_allocation_with_integral_cost(time_target, total_mass):
    final_marginal_discount = 0.0 
    
    for i in range(10):
        avg_discount = final_marginal_discount * 0.5
        effective_base_costs = df_bases["Cost"] * (1 - avg_discount)
        avg_rocket_cost = effective_base_costs.mean()
        
        rem = total_mass
        cost_iter = 0
        mass_ele = 0
        mass_roc = 0
        
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
            for idx, b in df_bases.iterrows():
                if needed_rocket <= 0: break
                can_take = min(needed_rocket, b["Yearly"] * time_target)
                cost_iter += can_take * b["Cost"] * (1 - avg_discount)
                mass_roc += can_take
                needed_rocket -= can_take
        
        new_marginal_discount = SCALE_DISCOUNT * (mass_roc / total_mass)
        if abs(new_marginal_discount - final_marginal_discount) < 0.0001:
            return cost_iter, mass_roc, mass_ele, new_marginal_discount
        final_marginal_discount = new_marginal_discount
        
    return cost_iter, mass_roc, mass_ele, final_marginal_discount

# [新增] 离散事件仿真器 (集成：飞行失败风险逻辑)
def simulate_rocket_trajectory_with_failure(target_mass, seed=42, pace_years=None, p_fail_override=None):
    if target_mass <= 0:
        return np.array([0]), np.array([0]), np.array([0])
        
    np.random.seed(seed)
    # [修改] 使用覆盖值或全局常量
    p_fail = p_fail_override if p_fail_override is not None else P_FAIL_CONST
    
    num_bases = len(df_bases)
    daily_caps = df_bases["Yearly"].values / 365.0 
    unit_costs = df_bases["Cost"].values
    
    # 配速逻辑
    if pace_years is not None:
        daily_needed_raw = target_mass / (pace_years * 365.0)
        # 此时不再进行补偿系数计算 (不再预知失败)，按照 150 年的原始需求去推
        daily_needed = daily_needed_raw 
        
        daily_max_possible = np.sum(daily_caps)
        if daily_max_possible > daily_needed:
            throttle_factor = daily_needed / daily_max_possible
            daily_caps = daily_caps * throttle_factor

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
            prod = daily_caps[i]
            if current_mass + daily_mass_this_day + prod > target_mass:
                prod = target_mass - (current_mass + daily_mass_this_day)
            
            if prod <= 0: continue

            # 折扣计算
            midpoint_mass = current_mass + daily_mass_this_day + (prod / 2.0)
            # 失败时的折扣近似用当前的
            discount_fail = SCALE_DISCOUNT * ((current_mass + daily_mass_this_day) / M_TOTAL)
            
            base_launch_cost = prod * unit_costs[i] 

            # >>>>>>>>> 判定: 炸了吗？ (新增逻辑) <<<<<<<<<
            if np.random.rand() < p_fail:
                # === 炸了 (Failure) ===
                # 损失 = 火箭发射成本(含折扣) + 货物成本(全额)
                launch_loss = base_launch_cost * (1 - discount_fail)
                payload_loss = prod * C_PAYLOAD_PER_TON
                
                total_acc_cost += (launch_loss + payload_loss)
                # 不增加运量 (current_mass 不变)
                continue
            
            else:
                # === 成功 (Success) ===
                discount_success = SCALE_DISCOUNT * (midpoint_mass / M_TOTAL)
                total_acc_cost += prod * unit_costs[i] * (1 - discount_success)
                daily_mass_this_day += prod

        current_mass += daily_mass_this_day
        
        if day_count % 30 == 0 or current_mass >= target_mass:
            days.append(day_count / 365.0)
            masses.append(current_mass)
            costs.append(total_acc_cost / 1e12)
            
        if day_count > 2000 * 365: break 
        
    return np.array(days), np.array(costs), np.array(masses)

# =============================================================================
# 3. 数据生成
# =============================================================================
print("正在执行计算 (Scenario 3: Launch Failure with Sensitivity)...")

# [新增] 敏感性分析采样
np.random.seed(2026)
p_fail_samples = np.random.normal(P_FAIL_CONST, P_FAIL_STD, 10)
p_fail_samples = np.clip(p_fail_samples, 0.001, 0.05) # 限制在合理区间

# --- A. 帕累托前沿计算 ---
t_axis = np.linspace(T_MIN, T_MAX, 100)
pareto_data = []

for T in t_axis:
    raw_cost, m_roc, m_ele, avg_discount = solve_allocation_with_integral_cost(T, M_TOTAL)
    pressure = (T_MAX - T) / (T_MAX - T_MIN)
    final_cost = raw_cost * (1 + 0.5 * (pressure ** 2))
    
    pareto_data.append({
        "Time (Years)": T, 
        "Total Cost (Trillion USD)": final_cost / 1e12
    })

df_pareto = pd.DataFrame(pareto_data)

# --- B. 关键点 (M1, M2, M3) 计算 [修改部分: 使用仿真器体现炸机影响] ---

# Method 1 (Elevator) - [不受炸机影响]
t_m1 = T_MAX
m_ele_m1, m_roc_m1 = M_TOTAL, 0
cost_m1 = (M_TOTAL * cost_g) / 1e12 

# Method 2 (Rocket) - [修改: 使用仿真器并添加敏感性采样]
print("  Simulating M2 (Rocket) with Sensitivity...")
m2_results = []
for p in p_fail_samples:
    d, c, _ = simulate_rocket_trajectory_with_failure(M_TOTAL, seed=101, p_fail_override=p)
    m2_results.append((d, c))

# 基础显示用 (均值/中位数采样)
m2_days, m2_costs = m2_results[0] 
t_m2 = m2_days[-1] 
cost_m2_raw = m2_costs[-1]
pressure_m2 = (T_MAX - t_m2) / (T_MAX - T_MIN)
final_cost_m2 = cost_m2_raw * 1e12 * (1 + 0.5 * pressure_m2**2)
cost_m2 = final_cost_m2 / 1e12
years_m2 = m2_days + START_YEAR
costs_m2 = m2_costs * (final_cost_m2 / (cost_m2_raw * 1e12))


# Method 3 (Hybrid) - [修改: 移除预知未来的积分修正并添加敏感性采样]
print("  Simulating M3 (Hybrid) with Sensitivity...")
t_m3_plan = 150.0  # 我们设定的计划时间
_, m_roc_plan, m_ele_plan, _ = solve_allocation_with_integral_cost(t_m3_plan, M_TOTAL)

m3_results = []
for p in p_fail_samples:
    # 电梯部分(理想)
    y_ele = np.linspace(0, t_m3_plan, 1000)
    c_ele = y_ele * (m_ele_plan * cost_g / t_m3_plan) / 1e12
    # 火箭部分(带风险)
    d_roc, c_roc, _ = simulate_rocket_trajectory_with_failure(m_roc_plan, seed=101, pace_years=t_m3_plan, p_fail_override=p)
    
    t_real = max(t_m3_plan, d_roc[-1])
    common_t = np.linspace(0, t_real, 1000)
    i_roc = np.interp(common_t, d_roc, c_roc)
    i_ele = np.interp(common_t, y_ele, c_ele)
    i_ele[common_t > t_m3_plan] = c_ele[-1]
    
    total_raw = (i_roc + i_ele)
    p_m3 = (T_MAX - t_real) / (T_MAX - T_MIN)
    f_c3 = total_raw[-1] * (1 + 0.5 * p_m3**2)
    m3_results.append((common_t + START_YEAR, total_raw * (f_c3 / total_raw[-1])))

# 基础显示用
years_m3, costs_m3 = m3_results[0]
t_m3 = years_m3[-1] - START_YEAR
cost_m3 = costs_m3[-1]

# --- C. 仿真曲线数据生成 (对于 M1 保持基底函数的占位) ---
def calculate_simulation_data(duration, target_total_cost, m_ele_total, m_roc_total):
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
        if M_TOTAL > 0:
            progress = cumulative_roc_mass / M_TOTAL
        else:
            progress = 0
        current_discount = SCALE_DISCOUNT * progress
        current_rocket_price = avg_base_price * (1 - current_discount)
        cost_step_roc = step_roc * current_rocket_price
        cumulative_cost += (cost_step_ele + cost_step_roc)
        cumulative_roc_mass += step_roc
        
    if cumulative_cost > 0:
        scale_fix = target_total_cost / cumulative_cost
    else:
        scale_fix = 1.0
    final_costs = [c * scale_fix for c in costs]
    return years + START_YEAR, final_costs

years_m1, costs_m1 = calculate_simulation_data(t_m1, cost_m1 * 1e12, M_TOTAL, 0)

# =============================================================================
# 4. 绘图 1: 帕累托前沿
# =============================================================================
plt.figure(figsize=(12, 7))

plt.plot(df_pareto["Time (Years)"], df_pareto["Total Cost (Trillion USD)"], 'b-', linewidth=3, label='Method 3: Hybrid Pareto (Integral Cost)')
plt.fill_between(df_pareto["Time (Years)"], df_pareto["Total Cost (Trillion USD)"], color='blue', alpha=0.1)

plt.scatter(t_m1, cost_m1, color='green', s=120, zorder=5, label='Method 1: Elevator Only')
plt.annotate(f'M1\n({t_m1:.1f}y, ${cost_m1:.1f}T)', (t_m1, cost_m1), 
              xytext=(-40, 10), textcoords='offset points', color='green', fontweight='bold')

plt.scatter(t_m2, cost_m2, color='red', s=120, zorder=5, label='Method 2: Rockets Only')
plt.annotate(f'M2\n({t_m2:.1f}y, ${cost_m2:.1f}T)', (t_m2, cost_m2), 
              xytext=(20, 10), textcoords='offset points', color='red', fontweight='bold')

closest_idx = (np.abs(df_pareto['Time (Years)'] - t_m2)).argmin()
hybrid_cost_at_m2 = df_pareto.iloc[closest_idx]['Total Cost (Trillion USD)']
plt.vlines(t_m2, hybrid_cost_at_m2, cost_m2, colors='red', linestyles=':', alpha=0.5)
plt.text(t_m2, (cost_m2 + hybrid_cost_at_m2)/2, "  Efficiency Gap", color='red', fontsize=9, rotation=90, va='center')

plt.title("Advanced Pareto Frontier: Integral Cost Calculation (True Learning Curve)", fontsize=14)
plt.xlabel("Completion Time (Years)", fontsize=12)
plt.ylabel("Total Project Cost (Trillion USD)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig("pareto_frontier_scenario3.png")
plt.show()

# =============================================================================
# 5. 绘图 2: 成本累积仿真
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

# Method 1
ax.plot(years_m1, costs_m1, color=colors['elevator'], linestyle='-', linewidth=2.5, label='Method 1: Elevator Only')
ax.scatter(years_m1[-1], costs_m1[-1], color=colors['elevator'], s=80, zorder=5, edgecolor='white', linewidth=1.5)

# Method 2 + Sensitivity Range
m2_all_costs = [np.interp(years_m2, r[0]+START_YEAR, r[1]) for r in m2_results]
ax.fill_between(years_m2, np.min(m2_all_costs, axis=0), np.max(m2_all_costs, axis=0), color=colors['rocket'], alpha=0.15)
ax.plot(years_m2, costs_m2, color=colors['rocket'], linestyle='-', linewidth=2.5, label='Method 2: Rockets Only (Mean)')
ax.scatter(years_m2[-1], costs_m2[-1], color=colors['rocket'], s=80, zorder=5, edgecolor='white', linewidth=1.5)

# Method 3 + Sensitivity Range
m3_all_costs = [np.interp(years_m3, r[0], r[1]) for r in m3_results]
ax.fill_between(years_m3, np.min(m3_all_costs, axis=0), np.max(m3_all_costs, axis=0), color=colors['hybrid'], alpha=0.15)
ax.plot(years_m3, costs_m3, color=colors['hybrid'], linestyle='-', linewidth=2.5, label='Method 3: Hybrid Strategy')
ax.scatter(years_m3[-1], costs_m3[-1], color=colors['hybrid'], s=80, zorder=5, edgecolor='white', linewidth=1.5)

# Annotations
ax.annotate(f'{int(years_m1[-1])}\n${costs_m1[-1]:.1f}T', (years_m1[-1], costs_m1[-1]), xytext=(-40, 10), textcoords="offset points", color=colors['elevator'], fontweight='bold', fontsize=9)
ax.annotate(f'{int(years_m2[-1])}\n${costs_m2[-1]:.1f}T', (years_m2[-1], costs_m2[-1]), xytext=(10, 5), textcoords="offset points", color=colors['rocket'], fontweight='bold', fontsize=9)
ax.annotate(f'{int(years_m3[-1])}\n${costs_m3[-1]:.1f}T', (years_m3[-1], costs_m3[-1]), xytext=(10, 5), textcoords="offset points", color=colors['hybrid'], fontweight='bold', fontsize=9)

ax.set_title("Cumulative Cost Trajectories: Risk & Learning Sensitivity", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Accumulated Cost (Trillion USD)", fontsize=12)
ax.grid(True, which='major', linestyle='-', alpha=0.3, color='gray')
ax.minorticks_on()
ax.set_xlim(START_YEAR, START_YEAR + 450)
ax.set_ylim(0, max(final_cost_m2, costs_m3[-1]*1e12, cost_m1*1e12)/1e12 * 1.1)
ax.legend(loc='upper left', frameon=False, fontsize=11)

# Inset Zoom
axins = inset_axes(ax, width="40%", height="40%", loc=1, borderpad=3)
axins.plot(years_m1, costs_m1, color=colors['elevator'], linewidth=2)
axins.plot(years_m2, costs_m2, color=colors['rocket'], linewidth=2)
axins.fill_between(years_m2, np.min(m2_all_costs, axis=0), np.max(m2_all_costs, axis=0), color=colors['rocket'], alpha=0.1)
axins.plot(years_m3, costs_m3, color=colors['hybrid'], linewidth=2)
axins.fill_between(years_m3, np.min(m3_all_costs, axis=0), np.max(m3_all_costs, axis=0), color=colors['hybrid'], alpha=0.1)
axins.set_xlim(2050, 2250)
axins.set_ylim(0, final_cost_m2/1e12 * 1.2)
axins.grid(True, linestyle=':', alpha=0.4)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--")

plt.tight_layout()
plt.savefig("simulation_trajectories_scenario3.png", dpi=300)
plt.show()

print("--- 所有敏感性分析图表绘制完成 ---")