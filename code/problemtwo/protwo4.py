import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# =============================================================================
# 1. 基础参数与设置
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
SCALE_DISCOUNT = 0.2  # 20% 规模折扣
START_YEAR = 2050

# [新增] 情况四：电梯故障参数
MTBF_ELE = 300.0             # 平均故障间隔 (天)
MTTR_ELE = 7.0             # 平均维修时间 (天)

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

# [新增] 辅助函数：计算火箭成本积分 (含规模效应)
def calculate_rocket_cost_integral(mass_start, mass_end, total_project_mass):
    # 计算加权平均基础单价
    avg_base_price = 0
    total_cap = df_bases["Yearly"].sum()
    for _, b in df_bases.iterrows():
        avg_base_price += b["Cost"] * (b["Yearly"]/total_cap)
    
    # 积分公式: Cost = P * (m - S * m^2 / (2 * M_total))
    def integral_func(m):
        return avg_base_price * (m - SCALE_DISCOUNT * (m**2) / (2 * total_project_mass))
    
    return integral_func(mass_end) - integral_func(mass_start)

# [新增] 辅助函数：电梯离散仿真 (含故障逻辑)
def simulate_elevator_with_breakdowns(target_mass, time_limit_years, unit_cost, seed=42):
    np.random.seed(seed)
    daily_nominal = yearly_payload_g / 365.0
    
    # 如果没有指定质量目标，就跑到时间结束；如果指定了，就跑到运完
    max_days = int(time_limit_years * 365) if time_limit_years else int(1e6) 
    
    current_mass = 0.0
    total_cost = 0.0
    
    days = [0]
    costs = [0]
    masses = [0]
    
    repair_days_remaining = 0
    
    for day in range(1, max_days + 1):
        if repair_days_remaining > 0:
            # 故障中: 成本不增加(不产生运费), 运量不增加, 只是时间流逝
            repair_days_remaining -= 1
        else:
            # 运行中
            if np.random.rand() < (1.0 / MTBF_ELE):
                # 坏了: 无成本增加(简化), 仅触发维修时间
                repair_days_remaining = int(np.random.normal(MTTR_ELE, 2))
                if repair_days_remaining < 1: repair_days_remaining = 1
            else:
                # 正常运货
                prod = daily_nominal
                # 如果有目标质量，进行截断
                if target_mass and (current_mass + prod > target_mass):
                    prod = target_mass - current_mass
                
                total_cost += prod * unit_cost
                current_mass += prod
        
        # 记录数据
        if day % 30 == 0:
            days.append(day / 365.0)
            costs.append(total_cost / 1e12)
            masses.append(current_mass)
            
        # 如果设定了目标质量且完成了，退出
        if target_mass and current_mass >= target_mass:
            break
            
    return np.array(days), np.array(costs), np.array(masses)

# =============================================================================
# 3. 数据生成
# =============================================================================
print("正在执行计算 (Scenario 4: Elevator Breakdown & Rocket Backfill)...")

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


# --- B. 关键点 (M1, M2, M3) 计算 [修改部分] ---

# Method 1 (Elevator Only) - [修改: 使用仿真体现故障影响]
print("  Simulating M1...")
# 仿真直到运完 M_TOTAL
m1_days, m1_costs, _ = simulate_elevator_with_breakdowns(M_TOTAL, None, cost_g, seed=2024)
t_m1 = m1_days[-1]         # 实际完工时间 (会比理想 T_MAX 长)
cost_m1 = m1_costs[-1]     # 实际成本
years_m1 = m1_days + START_YEAR
costs_m1 = m1_costs


# Method 2 (Rocket Only) - [保持基底逻辑: 理想火箭]
t_m2 = M_TOTAL / total_yearly_rockets
# 计算积分成本
m2_cost_val = calculate_rocket_cost_integral(0, M_TOTAL, M_TOTAL)
# 加上压力惩罚 (保持画图逻辑一致)
pressure_m2 = (T_MAX - t_m2) / (T_MAX - T_MIN)
final_cost_m2 = m2_cost_val * (1 + 0.5 * pressure_m2**2)
cost_m2 = final_cost_m2 / 1e12
# 生成曲线
years_m2 = np.linspace(0, t_m2, 200)
masses_m2 = years_m2 * (M_TOTAL / t_m2)
costs_m2 = []
for m in masses_m2:
    c = calculate_rocket_cost_integral(0, m, M_TOTAL)
    costs_m2.append(c / 1e12 * (final_cost_m2/m2_cost_val)) # 修正压力系数
years_m2 += START_YEAR
costs_m2 = np.array(costs_m2)


# Method 3 (Hybrid) - [核心修改: 动态救火逻辑]
print("  Simulating M3 (Backfill)...")
t_m3 = 150.0  # 目标 150 年

# 1. 规划 (Planning): 盲目乐观，认为电梯不坏
_, m_roc_plan, m_ele_plan, _ = solve_allocation_with_integral_cost(t_m3, M_TOTAL)

# 2. 电梯执行 (Execution): 带着故障跑 150 年
m3_ele_days, m3_ele_costs, m3_ele_masses = simulate_elevator_with_breakdowns(m_ele_plan, t_m3, cost_g, seed=2025)
m_ele_actual = m3_ele_masses[-1]

# 3. 计算缺口 (The Gap)
gap = m_ele_plan - m_ele_actual
print(f"    - Planned Elevator: {m_ele_plan/1e6:.2f} Mt")
print(f"    - Actual Elevator:  {m_ele_actual/1e6:.2f} Mt")
print(f"    - Gap to Backfill:  {gap/1e6:.2f} Mt")

# 4. 火箭兜底 (Rocket Backfill)
# 火箭总任务 = 原计划 + 缺口
m_roc_total_needed = m_roc_plan + gap
# 生成火箭成本曲线 (假设火箭在 150 年内均匀加速完成了这个更大的总量)
m3_roc_costs = []
# 将火箭进度对齐到电梯的时间轴
m3_roc_masses_trajectory = m3_ele_days * (m_roc_total_needed / t_m3) 

for m in m3_roc_masses_trajectory:
    # 规模效应核心: 这里的 m 是包含 gap 的，积分计算会自动处理 scale discount
    c = calculate_rocket_cost_integral(0, m, M_TOTAL)
    m3_roc_costs.append(c / 1e12)
m3_roc_costs = np.array(m3_roc_costs)

# 5. 合并成本
# 对齐长度 (防止仿真步长不一致)
min_len = min(len(m3_ele_costs), len(m3_roc_costs))
total_m3_cost_curve = m3_ele_costs[:min_len] + m3_roc_costs[:min_len]
years_m3 = m3_ele_days[:min_len] + START_YEAR
costs_m3 = total_m3_cost_curve

# 6. 计算最终点 (用于画图标记)
raw_cost_m3 = total_m3_cost_curve[-1] * 1e12
pressure_m3 = (T_MAX - t_m3) / (T_MAX - T_MIN)
final_cost_m3 = raw_cost_m3 * (1 + 0.5 * pressure_m3**2)
cost_m3 = final_cost_m3 / 1e12
costs_m3 = costs_m3 * (final_cost_m3 / (raw_cost_m3 if raw_cost_m3 > 0 else 1)) # 压力修正

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
plt.savefig("pareto_frontier_scenario4.png")
plt.show()

# =============================================================================
# 5. 绘图 2: 成本累积仿真
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(years_m1, costs_m1, color=colors['elevator'], linestyle='-', linewidth=2.5, label='Method 1: Elevator Only')
ax.scatter(years_m1[-1], costs_m1[-1], color=colors['elevator'], s=80, zorder=5, edgecolor='white', linewidth=1.5)
ax.annotate(f'{int(years_m1[-1])}\n${costs_m1[-1]:.1f}T', (years_m1[-1], costs_m1[-1]), 
             xytext=(-40, 10), textcoords="offset points", 
             color=colors['elevator'], fontweight='bold', fontsize=9, va='center')

ax.plot(years_m2, costs_m2, color=colors['rocket'], linestyle='-', linewidth=2.5, label='Method 2: Rockets Only')
ax.scatter(years_m2[-1], costs_m2[-1], color=colors['rocket'], s=80, zorder=5, edgecolor='white', linewidth=1.5)
ax.annotate(f'{int(years_m2[-1])}\n${costs_m2[-1]:.1f}T', (years_m2[-1], costs_m2[-1]), 
             xytext=(10, 5), textcoords="offset points", 
             color=colors['rocket'], fontweight='bold', fontsize=9, va='center')

ax.plot(years_m3, costs_m3, color=colors['hybrid'], linestyle='-', linewidth=2.5, label='Method 3: Hybrid Strategy')
ax.scatter(years_m3[-1], costs_m3[-1], color=colors['hybrid'], s=80, zorder=5, edgecolor='white', linewidth=1.5)
ax.annotate(f'{int(years_m3[-1])}\n${costs_m3[-1]:.1f}T', (years_m3[-1], costs_m3[-1]), 
             xytext=(10, 5), textcoords="offset points", 
             color=colors['hybrid'], fontweight='bold', fontsize=9, va='center')

ax.set_title("Cumulative Cost Trajectories: Learning Curve Impact", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Accumulated Cost (Trillion USD)", fontsize=12)
ax.grid(True, which='major', linestyle='-', alpha=0.3, color='gray')
ax.grid(True, which='minor', linestyle=':', alpha=0.15, color='gray')
ax.minorticks_on()
ax.set_xlim(START_YEAR, START_YEAR + 450)
ax.set_ylim(0, max(final_cost_m2, final_cost_m3, cost_m1*1e12)/1e12 * 1.1)
ax.legend(loc='upper left', frameon=False, fontsize=11)
ax.axvspan(2050, 2200, color='gray', alpha=0.05)
ax.text(2125, ax.get_ylim()[1]*0.05, "Active Phase", ha='center', color='gray', fontsize=10, alpha=0.6)

# Inset Zoom
axins = inset_axes(ax, width="40%", height="40%", loc=1, borderpad=3)
axins.plot(years_m1, costs_m1, color=colors['elevator'], linestyle='-', linewidth=2.5)
axins.plot(years_m2, costs_m2, color=colors['rocket'], linestyle='-', linewidth=2.5)
axins.plot(years_m3, costs_m3, color=colors['hybrid'], linestyle='-', linewidth=2.5)
axins.set_xlim(2050, 2220)
axins.set_ylim(0, final_cost_m2/1e12 * 1.1)
axins.grid(True, linestyle=':', alpha=0.4)
axins.set_title("Zoom: 2050-2220 (Curvature Detail)", fontsize=9)
axins.tick_params(labelsize=8)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--")

plt.tight_layout()
plt.savefig("simulation_trajectories_scenario4.png", dpi=300)
plt.show()

print("--- 所有图表绘制完成 (无Excel导出) ---")