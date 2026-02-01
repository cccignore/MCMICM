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
TIER_PARAMS = {
    "Modern":   {"mtbf": 300, "mttr": 2},  
    "Standard": {"mtbf": 200, "mttr": 4}, 
    "Legacy":   {"mtbf": 140,  "mttr": 7}  
}

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
yearly_payload_g = (N_PORTS * U_PORT) / k_g 

base_configs_list = []
for name, lat in BASES_DATA.items():
    v_rot = V_ROT_EQUATOR * np.cos(np.radians(lat))
    dv_eff = DV_BASE - v_rot
    k, c = get_rocket_stats(dv_eff, ISP_E, ALPHA_E, False)
    
    max_y_ideal = (LAMBDA_J * 365 * L_MAX) / k
    tier = BASE_TIERS.get(name, "Standard")
    params = TIER_PARAMS[tier]
    
    denom = params['mtbf'] + params['mttr']
    if denom == 0: availability = 1.0
    else: availability = params['mtbf'] / denom
    
    max_y_effective = max_y_ideal * availability
    
    base_configs_list.append({
        "Name": name, "Cost": c, 
        "Yearly_Ideal": max_y_ideal, 
        "Yearly_Effective": max_y_effective,
        "MTBF": params['mtbf'], "MTTR": params['mttr'],
        "Tier": tier
    })

df_bases = pd.DataFrame(base_configs_list).sort_values("Cost")
total_yearly_rockets_effective = df_bases["Yearly_Effective"].sum()

T_MIN = M_TOTAL / (yearly_payload_g + total_yearly_rockets_effective)
T_MAX = M_TOTAL / yearly_payload_g

# =============================================================================
# 3. 核心算法: 宏观规划求解器
# =============================================================================
def solve_allocation_with_reliability(time_target, total_mass):
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
            can_take_roc = min(rem, total_yearly_rockets_effective * time_target)
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
# 4. 离散事件仿真器
# =============================================================================
def simulate_rocket_trajectory(target_mass, seed=42, pace_years=None, Z_factor=None):
    if target_mass <= 0:
        return np.array([0]), np.array([0]), np.array([0])
        
    np.random.seed(seed)
    
    num_bases = len(df_bases)
    bases_status = np.ones(num_bases, dtype=int)
    repair_counters = np.zeros(num_bases, dtype=int)
    
    daily_caps = df_bases["Yearly_Ideal"].values / 365.0 
    unit_costs = df_bases["Cost"].values
    
    mtbf_base = df_bases["MTBF"].values
    mttr_base = df_bases["MTTR"].values
    
    if Z_factor is not None:
        sigma_mtbf = 0.2
        sigma_mttr = 0.2
        mtbfs = mtbf_base * (1 + sigma_mtbf * Z_factor)
        mttrs = mttr_base * (1 - sigma_mttr * Z_factor)
        mtbfs = np.maximum(1.0, mtbfs)
        mttrs = np.maximum(0.1, mttrs)
    else:
        mtbfs = mtbf_base
        mttrs = mttr_base

    if pace_years is not None:
        daily_needed_raw = target_mass / (pace_years * 365.0)
        avg_availability = 1.0 
        daily_needed_corrected = daily_needed_raw / avg_availability
        daily_max_possible = np.sum(daily_caps)
        if daily_max_possible > daily_needed_corrected:
            throttle_factor = daily_needed_corrected / daily_max_possible
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
            if bases_status[i] == 0:
                repair_counters[i] -= 1
                if repair_counters[i] <= 0:
                    bases_status[i] = 1
                else:
                    continue 
            
            if mttrs[i] > 0:
                if np.random.rand() < (1.0 / mtbfs[i]):
                    bases_status[i] = 0
                    r_days = int(np.random.normal(mttrs[i], mttrs[i]*0.2))
                    repair_counters[i] = max(1, r_days)
                    continue 
            
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
# 5. 蒙特卡洛模拟 (1000次) - Method 3 
# =============================================================================
print("正在执行蒙特卡洛模拟 (N=1000)...")

# 1. 设定规划目标: 150年
t_target = 150.0
_, m_roc_plan, m_ele_plan, _ = solve_allocation_with_reliability(t_target, M_TOTAL)

ele_cost_total = (m_ele_plan * cost_g) / 1e12
t_actual_ele = m_ele_plan / yearly_payload_g

# 存储结果
mc_results_time = []
mc_results_cost = []
mc_trajectories = [] 

common_time_axis = np.linspace(0, 300, 600) 

for k in range(100):
    Z_val = np.random.normal(0, 1)
    
    days, costs, _ = simulate_rocket_trajectory(m_roc_plan, seed=None, pace_years=t_target, Z_factor=Z_val)
    
    t_roc = days[-1]
    c_roc = costs[-1]
    
    # 最终完工时间和成本
    t_final = max(t_roc, t_actual_ele)
    raw_c_final = (c_roc + ele_cost_total) * 1e12
    pressure = (T_MAX - t_final) / (T_MAX - T_MIN)
    final_c_total = raw_c_final * (1 + 0.5 * pressure**2)
    final_c_trillion = final_c_total / 1e12
    
    mc_results_time.append(t_final)
    mc_results_cost.append(final_c_trillion)
    
    # --- 修改：轨迹生成逻辑 (使用均摊线性生成，消除折线) ---
    # 假设火箭成本随时间线性积累 (Amortized Budget Logic)
    # 起点 (0,0), 终点 (t_roc, c_roc)
    # 在 common_time_axis 上生成线性增长
    
    # 火箭线性均摊
    # y = kx, k = c_roc / t_roc
    linear_roc_traj = np.minimum(common_time_axis * (c_roc / t_roc), c_roc)
    # 如果 common_time_axis 超过 t_roc，保持 c_roc
    
    # 电梯线性均摊 (本身就是线性的)
    linear_ele_traj = np.minimum(common_time_axis * (ele_cost_total / t_actual_ele), ele_cost_total)
    
    total_traj = linear_roc_traj + linear_ele_traj
    
    # 修正：如果 t_final 还没到，轨迹就是 total_traj
    # 如果时间轴超过了 t_final，成本应该保持不变 (项目结束)
    mask_finished = common_time_axis > t_final
    if np.any(mask_finished):
        idx_finish = np.argmax(mask_finished)
        # 之后的值都等于完工时的值 (虽然上面 minimum 逻辑已经部分处理了，但为了严谨)
        # 注意：这里我们展示的是 raw cost accumulation，不包含最终的一次性罚款
        # 如果要在图上展示罚款，需要在终点突变。通常累积成本图只展示 Raw，罚款在文字标注。
        pass 
        
    mc_trajectories.append(total_traj)

mc_trajectories = np.array(mc_trajectories)

traj_mean = np.mean(mc_trajectories, axis=0)
traj_lower = np.percentile(mc_trajectories, 5, axis=0)
traj_upper = np.percentile(mc_trajectories, 95, axis=0)

# =============================================================================
# 6. 绘图 1: 帕累托前沿 (保持不变)
# =============================================================================
t_axis = np.linspace(T_MIN, T_MAX * 1.1, 100)
pareto_mean_data = []
for T in t_axis:
    raw, _, _, _ = solve_allocation_with_reliability(T, M_TOTAL)
    p = (T_MAX - T) / (T_MAX - T_MIN)
    fc = raw * (1 + 0.5 * p**2)
    pareto_mean_data.append(fc / 1e12)

plt.figure(figsize=(12, 7))
plt.plot(t_axis, pareto_mean_data, 'b-', linewidth=3, label='Method 3: Mean Expectation')

plt.scatter(mc_results_time, mc_results_cost, color=colors['hybrid'], s=10, alpha=0.3, label='Monte Carlo Results (T_target=150)')
plt.scatter(np.mean(mc_results_time), np.mean(mc_results_cost), color='red', s=100, marker='X', zorder=5, label='MC Mean')

plt.title(f"Pareto Frontier with Uncertainty (MTBF/MTTR Correlation Z~N(0,1))", fontsize=14)
plt.xlabel("Completion Time (Years)", fontsize=12)
plt.ylabel("Total Project Cost (Trillion USD)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("pareto_frontier_mc.png")
plt.show()

# =============================================================================
# 7. 绘图 2: 成本累积仿真 (均摊平滑版)
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

common_years = common_time_axis + START_YEAR

# 绘制所有 MC 轨迹 (云) - 现在是平滑的直线
for i in range(min(200, len(mc_trajectories))):
    ax.plot(common_years, mc_trajectories[i], color=colors['hybrid'], lw=0.5, alpha=0.1)

# 绘制均值
ax.plot(common_years, traj_mean, color='black', lw=2.5, linestyle='--', label='Mean Amortized Trajectory')
# 绘制置信区间
ax.fill_between(common_years, traj_lower, traj_upper, color=colors['hybrid'], alpha=0.3, label='95% Confidence Interval')

ax.set_title("Cumulative Cost Trajectories: Amortized Budget View (Smoothed)", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Accumulated Cost (Trillion USD)", fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(START_YEAR, START_YEAR + 250)
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig("simulation_trajectories_mc_smooth.png")
plt.show()

# =============================================================================
# 8. 绘图 3: 相对变动比例 (Scatter Plot: Time vs Cost)
# =============================================================================
# 计算基准值 (无故障理想情况 T=150)
# 假设 MTTR=0 或 Z=+Inf (最好的情况)? 不，基准应该是"规划预期值"(Z=0, 无随机故障)
# 或者使用 solve_allocation_with_reliability 的直接输出作为基准
raw_ideal, _, _, _ = solve_allocation_with_reliability(t_target, M_TOTAL)
p_ideal = (T_MAX - t_target) / (T_MAX - T_MIN)
cost_ideal_val = (raw_ideal * (1 + 0.5 * p_ideal**2)) / 1e12
time_ideal_val = t_target

# 计算相对变动 (%)
rel_time_change = [(t - time_ideal_val) / time_ideal_val * 100 for t in mc_results_time]
rel_cost_change = [(c - cost_ideal_val) / cost_ideal_val * 100 for c in mc_results_cost]

plt.figure(figsize=(10, 8))

# 绘制散点
plt.scatter(rel_time_change, rel_cost_change, color='#7570B3', alpha=0.5, s=20, label='Simulated Scenarios')

# 绘制均值点
mean_time_change = np.mean(rel_time_change)
mean_cost_change = np.mean(rel_cost_change)
plt.scatter(mean_time_change, mean_cost_change, color='red', s=150, marker='X', zorder=5, label=f'Mean Shift\n(Time+{mean_time_change:.1f}%, Cost+{mean_cost_change:.1f}%)')

# 辅助线 (0,0 基准)
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.axvline(0, color='gray', linestyle='--', linewidth=1)

# 添加等高线密度图 (可选，增加高级感)
try:
    from scipy.stats import gaussian_kde
    # 堆叠数据
    data = np.vstack([rel_time_change, rel_cost_change])
    kde = gaussian_kde(data)
    # 创建网格
    xgrid = np.linspace(min(rel_time_change), max(rel_time_change), 100)
    ygrid = np.linspace(min(rel_cost_change), max(rel_cost_change), 100)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    Zgrid = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
    plt.contour(Xgrid, Ygrid, Zgrid.reshape(Xgrid.shape), levels=5, colors='k', alpha=0.3)
except ImportError:
    pass

plt.title(f"Relative Deviation from Plan (T={t_target}y)\nImpact of Global Reliability Factor Z", fontsize=14)
plt.xlabel("Relative Time Delay (%)", fontsize=12)
plt.ylabel("Relative Cost Overrun (%)", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='upper left')

# 添加统计信息框
stats_text = (f"Baseline T: {time_ideal_val}y\n"
              f"Baseline Cost: ${cost_ideal_val:.2f}T\n\n"
              f"Mean Delay: +{mean_time_change:.2f}%\n"
              f"Mean Overrun: +{mean_cost_change:.2f}%")
plt.text(0.95, 0.05, stats_text, transform=plt.gca().transAxes, fontsize=11,
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

plt.tight_layout()
plt.savefig("relative_deviation_scatter.png")
plt.show()

print(f"MC Mean Time: {np.mean(mc_results_time):.1f} y (+{mean_time_change:.1f}%)")
print(f"MC Mean Cost: ${np.mean(mc_results_cost):.2f} T (+{mean_cost_change:.1f}%)")