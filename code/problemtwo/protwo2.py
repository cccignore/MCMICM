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
    "Modern":   {"mtbf": 300, "mttr": 2},  # 新建: 300天坏一次, 修5天
    "Standard": {"mtbf": 200, "mttr": 4}, # 标准: 150天坏一次, 修10天
    "Legacy":   {"mtbf": 140,  "mttr": 7}  # 老旧: 60天坏一次, 修20天 (瓶颈所在)
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
    max_y_effective = max_y_ideal * availability
    
    base_configs_list.append({
        "Name": name, "Cost": c, 
        "Yearly_Ideal": max_y_ideal, 
        "Yearly_Effective": max_y_effective, # 规划用这个
        "MTBF": params['mtbf'], "MTTR": params['mttr'],
        "Tier": tier
    })

df_bases = pd.DataFrame(base_configs_list).sort_values("Cost")
total_yearly_rockets_effective = df_bases["Yearly_Effective"].sum()

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
                can_take = min(needed_rocket, b["Yearly_Effective"] * time_target)
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
def simulate_rocket_trajectory(target_mass, seed=42, pace_years=None):
    """
    火箭故障仿真引擎
    修正点：
    1. mttr == 0 时完全绕过随机数逻辑。
    2. 成本累加与质量累加同步，消除步长导致的数值偏移。
    3. 增加 pace_years 参数，并自动计算 Availability 补偿，确保准时完工。
    """
    if target_mass <= 0:
        return np.array([0]), np.array([0]), np.array([0])
        
    np.random.seed(seed)
    
    num_bases = len(df_bases)
    bases_status = np.ones(num_bases, dtype=int)
    repair_counters = np.zeros(num_bases, dtype=int)
    
    daily_caps = df_bases["Yearly_Ideal"].values / 365.0 
    
    unit_costs = df_bases["Cost"].values
    mtbfs = df_bases["MTBF"].values
    mttrs = df_bases["MTTR"].values

    # --- 核心修改：智能配速逻辑 ---
    if pace_years is not None:
        # 1. 计算理论上每天需要运多少
        daily_needed_raw = target_mass / (pace_years * 365.0)
        
        # 2. 计算平均可用度 (Availability Correction)
        # 如果不补偿，故障期间的损失就会导致延误。我们通过"加班"来补偿。
        # 估算全局平均可用度 (保守估计)
        # 这里为了精确，我们取所有基地可用度的加权平均，或者简单取一个代表值
        # 简单的加权逻辑:
        avg_availability = 1.0
        total_mtbf = np.sum(mtbfs)
        total_mttr = np.sum(mttrs)
        if total_mttr > 0:
             # 这只是一个粗略的全局估算，用于调整配速
             # 为了更准，我们对每个基地分别调整，这里简化为全局提升速率
             avg_availability = np.mean(mtbfs / (mtbfs + mttrs))

        # 3. 修正后的每日目标 (Availability Adjusted Target)
        # "为了在停工期间也能扯平，我们在工作期间必须跑得更快"
        daily_needed_corrected = daily_needed_raw / avg_availability
        
        daily_max_possible = np.sum(daily_caps)
        
        # 如果修正后的需求还在物理极限内，就降速执行；否则全速跑(那就没办法了)
        if daily_max_possible > daily_needed_corrected:
            throttle_factor = daily_needed_corrected / daily_max_possible
            daily_caps = daily_caps * throttle_factor
        # ----------------------------
    
    days = [0]
    costs = [0]
    masses = [0]
    
    current_mass = 0.0
    total_acc_cost = 0.0  # 引入实时成本累加，消除步长误差
    
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
            
            # 2. 故障逻辑判定 (只有 MTTR > 0 才会触发损失)
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
            
            # --- 核心同步修正 ---
            # 1. 使用中点质量 (Midpoint Mass) 
            midpoint_mass = current_mass + daily_mass_this_day + (prod / 2.0)
            
            # 2. 瞬时折扣
            current_instant_discount = SCALE_DISCOUNT * (midpoint_mass / M_TOTAL)
            
            # 3. 累加成本
            total_acc_cost += prod * unit_costs[i] * (1 - current_instant_discount)
            
            daily_mass_this_day += prod
            
        current_mass += daily_mass_this_day
        
        # 每30天记录一次数据用于绘图
        if day_count % 30 == 0 or current_mass >= target_mass:
            days.append(day_count / 365.0)
            masses.append(current_mass)
            costs.append(total_acc_cost / 1e12)
            
        if day_count > 1000 * 365: break 
        
    return np.array(days), np.array(costs), np.array(masses)
# =============================================================================
# 5. 数据生成
# =============================================================================
print("正在执行异质性故障仿真...")

# --- A. 帕累托前沿 (基于期望有效运力) ---
t_axis = np.linspace(T_MIN, T_MAX * 1.1, 100) # 稍微延长以展示延误
pareto_data = []

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


# =============================================================================
# 6. 绘图 1: 帕累托前沿 (反映了可靠性惩罚)
# =============================================================================
plt.figure(figsize=(12, 7))

plt.plot(df_pareto["Time"], df_pareto["Cost"], 'b-', linewidth=3, label='Method 3: Hybrid (Reliability Adjusted)')
plt.fill_between(df_pareto["Time"], df_pareto["Cost"], color='blue', alpha=0.1)

plt.scatter(t_m1, cost_m1, color='green', s=120, zorder=5, label='Method 1: Elevator Only')
plt.annotate(f'M1\n({t_m1:.1f}y, ${cost_m1:.1f}T)', (t_m1, cost_m1), 
             xytext=(-40, 10), textcoords='offset points', color='green', fontweight='bold')

plt.scatter(t_m2_real, cost_m2, color='red', s=120, zorder=5, label='Method 2: Rockets (Real Stochastic)')
plt.annotate(f'M2\n({t_m2_real:.1f}y, ${cost_m2:.1f}T)', (t_m2_real, cost_m2), 
             xytext=(20, 10), textcoords='offset points', color='red', fontweight='bold')

# 标注火箭方案的延误
plt.arrow(t_m2_real - 20, cost_m2, 15, 0, head_width=2, head_length=3, fc='k', ec='k')
plt.text(t_m2_real - 25, cost_m2 + 5, "Delays due to\nLegacy Bases", ha='right', fontsize=9)

plt.title("Pareto Frontier: Impact of Infrastructure Reliability (MTBF/MTTR)", fontsize=14)
plt.xlabel("Completion Time (Years)", fontsize=12)
plt.ylabel("Total Project Cost (Trillion USD)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig("pareto_frontier_reliability.png")
plt.show()

# =============================================================================
# 7. 绘图 2: 成本累积仿真 (含随机性特征)
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

# M1
ax.plot(years_m1, costs_m1, color=colors['elevator'], linestyle='-', linewidth=2.5, label='Method 1: Elevator Only')
ax.scatter(years_m1[-1], costs_m1[-1], color=colors['elevator'], s=80, zorder=5, edgecolor='white', linewidth=1.5)
ax.annotate(f'{int(years_m1[-1])}\n${costs_m1[-1]:.1f}T', (years_m1[-1], costs_m1[-1]), 
             xytext=(-40, 10), textcoords="offset points", color=colors['elevator'], fontweight='bold', fontsize=9)

# M2 (Stochastic)
ax.plot(years_m2, m2_costs, color=colors['rocket'], linestyle='-', linewidth=2.5, label='Method 2: Rockets (Stochastic)')
ax.scatter(years_m2[-1], m2_costs[-1], color=colors['rocket'], s=80, zorder=5, edgecolor='white', linewidth=1.5)
ax.annotate(f'{int(years_m2[-1])}\n${m2_costs[-1]:.1f}T', (years_m2[-1], m2_costs[-1]), 
             xytext=(10, 5), textcoords="offset points", color=colors['rocket'], fontweight='bold', fontsize=9)

# M3 (Hybrid Stochastic)
ax.plot(years_m3, costs_m3, color=colors['hybrid'], linestyle='-', linewidth=2.5, label='Method 3: Hybrid Strategy')
ax.scatter(years_m3[-1], costs_m3[-1], color=colors['hybrid'], s=80, zorder=5, edgecolor='white', linewidth=1.5)
ax.annotate(f'{int(years_m3[-1])}\n${costs_m3[-1]:.1f}T', (years_m3[-1], costs_m3[-1]), 
             xytext=(10, 5), textcoords="offset points", color=colors['hybrid'], fontweight='bold', fontsize=9)

ax.set_title("Cumulative Cost Trajectories: Discrete Event Simulation of Failures", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Accumulated Cost (Trillion USD)", fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(START_YEAR, START_YEAR + 450)
ax.set_ylim(0, max(cost_m2, cost_m3_final, cost_m1)*1.1)
ax.legend(loc='upper left', frameon=False)
ax.axvspan(2050, 2200, color='gray', alpha=0.05)

# Inset Zoom
axins = inset_axes(ax, width="40%", height="40%", loc=1, borderpad=3)
axins.plot(years_m1, costs_m1, color=colors['elevator'], linestyle='-')
axins.plot(years_m2, m2_costs, color=colors['rocket'], linestyle='-')
axins.plot(years_m3, costs_m3, color=colors['hybrid'], linestyle='-')
axins.set_xlim(2050, 2220)
axins.set_ylim(0, cost_m2 * 1.1)
axins.grid(True, linestyle=':', alpha=0.4)
axins.set_title("Zoom: Stochastic Fluctuations (Step Effects)", fontsize=9)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--")

plt.tight_layout()
plt.savefig("simulation_trajectories_reliability.png", dpi=300)
plt.show()

print("--- 故障模型仿真完成 ---")
print(f"M2 实际完工时间: {t_m2_real:.1f} 年 (受老旧基地拖累)")
print(f"M3 实际完工时间: {t_m3_real:.1f} 年")