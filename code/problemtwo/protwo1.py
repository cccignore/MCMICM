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
# 2. 蒙特卡洛采样 (Monte Carlo) 与截断正态分布生成器 (修改模块)
# =============================================================================
def generate_monte_carlo_samples(n_samples=5000, mean=1.05, std=0.05, lower_bound=1.0):
    """
    使用高数量蒙特卡洛 (Monte Carlo) 方法生成截断正态分布样本 Y
    Y >= 1.0, 位于分母, Y越大运力越低
    """
    # 1. 直接生成大量正态分布样本 (为了应对截断，预生成 1.5 倍)
    raw_count = int(n_samples * 1.5)
    raw_samples = np.random.normal(mean, std, raw_count)
    
    # 2. 物理截断 (Reject samples < 1.0)
    # 这就是蒙特卡洛的本质：随机生成，不符合条件的扔掉
    valid_samples = raw_samples[raw_samples >= lower_bound]
    
    # 3. 如果有效样本不够，循环补充
    while len(valid_samples) < n_samples:
        more_samples = np.random.normal(mean, std, raw_count)
        valid_samples = np.concatenate([valid_samples, more_samples[more_samples >= lower_bound]])
    
    # 4. 取前 n_samples 个
    # 蒙特卡洛不需要排序或分层，保持随机性即可
    return valid_samples[:n_samples]

# =============================================================================
# 3. 核心计算函数
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
yearly_payload_g_nominal = (N_PORTS * U_PORT) / k_g # 标称年运力 (Y=1)

base_list = []
for name, lat in BASES_DATA.items():
    v_rot = V_ROT_EQUATOR * np.cos(np.radians(lat))
    dv_eff = DV_BASE - v_rot
    k, c = get_rocket_stats(dv_eff, ISP_E, ALPHA_E, False)
    max_y = (LAMBDA_J * 365 * L_MAX) / k
    base_list.append({"Cost": c, "Yearly": max_y, "Name": name})
df_bases = pd.DataFrame(base_list).sort_values("Cost")
total_yearly_rockets = df_bases["Yearly"].sum()
# 这里的 T_MIN 和 T_MAX 基于标称运力计算
T_MIN = M_TOTAL / (yearly_payload_g_nominal + total_yearly_rockets)
T_MAX_NOMINAL = M_TOTAL / yearly_payload_g_nominal

# 核心算法: 动态迭代求解器 (积分修正版)
# [修改] 增加参数 sway_Y, 默认为 1.0
def solve_allocation_with_integral_cost(time_target, total_mass, sway_Y=1.0):
    # 根据摇晃系数调整电梯年运力
    # 公式: Throughput = U / (Kg * Y) -> Nominal / Y
    current_yearly_elevator = yearly_payload_g_nominal / sway_Y
    
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
            # 极端情况优先火箭
            can_take_roc = min(rem, total_yearly_rockets * time_target)
            cost_iter += can_take_roc * avg_rocket_cost 
            mass_roc += can_take_roc
            rem -= can_take_roc
            cost_iter += rem * cost_g
            mass_ele += rem
        else:
            # 正常优先电梯 (使用受摇晃影响的运力)
            can_take_ele = min(rem, current_yearly_elevator * time_target)
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

# =============================================================================
# 4. 执行 高数量蒙特卡洛采样 与不确定性分析
# =============================================================================
print("执行 高数量蒙特卡洛采样 (Monte Carlo Simulation)...")
N_SAMPLES = 5000 # 设定为高数量 (从 50 增加到 5000)
sway_samples = generate_monte_carlo_samples(n_samples=N_SAMPLES, mean=1.05, std=0.05, lower_bound=1.0)

# 用于存储多次仿真的结果
pareto_runs = []    # 存储每次 Method 3 的帕累托曲线
m1_runs = []        # 存储每次 Method 1 的结果
# Method 2 不受摇晃影响，只算一次

# --- A. 批量计算 ---
# 1. Method 2 (Rocket Only) - 静态
t_m2 = M_TOTAL / total_yearly_rockets
c_m2_raw = 0
avg_discount_m2 = SCALE_DISCOUNT * 0.5
for _, b in df_bases.iterrows():
    c_m2_raw += b["Yearly"] * t_m2 * b["Cost"]
c_m2_raw *= (1 - avg_discount_m2)
pressure_m2 = (T_MAX_NOMINAL - t_m2) / (T_MAX_NOMINAL - T_MIN)
final_cost_m2 = c_m2_raw * (1 + 0.5 * pressure_m2**2)

# 2. 循环计算 Monte Carlo 样本 (Method 1 & 3)
t_axis = np.linspace(T_MIN, T_MAX_NOMINAL * 1.5, 100) # 稍微延长坐标轴以容纳更慢的情况

for Y_val in sway_samples:
    # --- Method 1 (Elevator Only) ---
    # 运力下降 -> 时间变长
    real_capacity = yearly_payload_g_nominal / Y_val
    t_m1_real = M_TOTAL / real_capacity
    # 成本不变 (假设按吨付费)，但可能因为时间延长导致项目管理惩罚? 
    # 这里保持原逻辑：成本 = M_TOTAL * cost_g (纯OpEx)
    cost_m1_real = (M_TOTAL * cost_g)
    m1_runs.append((t_m1_real, cost_m1_real))
    
    # --- Method 3 (Hybrid Pareto) ---
    run_costs = []
    run_times = []
    
    for T in t_axis:
        # 物理限制检查: 如果 T 太短，连全速都跑不完，成本无穷大
        max_possible = (real_capacity + total_yearly_rockets) * T
        if max_possible < M_TOTAL:
            run_costs.append(np.nan) # 不可行
            run_times.append(T)
            continue
            
        raw_cost, m_roc, m_ele, avg_d = solve_allocation_with_integral_cost(T, M_TOTAL, sway_Y=Y_val)
        
        # 压力惩罚: 使用标称的最慢时间作为基准，还是当前的?
        # 为了公平，统一使用标称 T_MAX_NOMINAL 作为项目预期的 Deadline
        pressure = max(0, (T_MAX_NOMINAL - T) / (T_MAX_NOMINAL - T_MIN))
        final_cost = raw_cost * (1 + 0.5 * (pressure ** 2))
        
        run_costs.append(final_cost / 1e12)
        run_times.append(T)
        
    pareto_runs.append(pd.DataFrame({"Time": run_times, "Cost": run_costs}))

# =============================================================================
# 5. 绘图 1: 帕累托前沿 (带置信区间)
# =============================================================================
plt.figure(figsize=(12, 7))

# 绘制 Method 3 的置信区间
# 我们需要对齐 Time 轴来计算 Cost 的分位数
# 创建一个统一的 DataFrame
df_concat = pd.concat(pareto_runs, axis=0)
# 按时间分组，计算均值和置信区间
grouped = df_concat.groupby("Time")["Cost"].agg(['mean', lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)]).reset_index()
grouped.columns = ["Time", "Mean", "Lower", "Upper"]
# 过滤掉 NaN (不可行区域)
grouped = grouped.dropna()



plt.plot(grouped["Time"], grouped["Mean"], color=colors['hybrid'], linewidth=3, label='Method 3: Hybrid (Mean)')
plt.fill_between(grouped["Time"], grouped["Lower"], grouped["Upper"], color=colors['hybrid'], alpha=0.2, label='95% Confidence Interval (Sway)')

# 绘制 Method 1 的散点云 (展示不确定性范围)
m1_times = [r[0] for r in m1_runs]
m1_costs = [r[1]/1e12 for r in m1_runs]
plt.scatter(m1_times, m1_costs, color=colors['elevator'], s=30, alpha=0.4, label='Method 1 Uncertainty Cloud')
# 标称点
plt.scatter(M_TOTAL/yearly_payload_g_nominal, (M_TOTAL*cost_g)/1e12, color='green', s=120, edgecolors='black', zorder=5, label='M1 Nominal')

# 绘制 Method 2 (固定点)
plt.scatter(t_m2, final_cost_m2/1e12, color='red', s=120, edgecolors='black', zorder=5, label='Method 2: Rockets Only (Robust)')

plt.title("Pareto Frontier under Elevator Sway Uncertainty (Monte Carlo, N=5000)", fontsize=14)
plt.xlabel("Completion Time (Years)", fontsize=12)
plt.ylabel("Total Project Cost (Trillion USD)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig("pareto_frontier_uncertainty.png")
plt.show()

# =============================================================================
# 6. 绘图 2: 成本累积仿真 (带置信区间)
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

# 辅助函数: 计算一条仿真曲线
def get_curve(duration, total_cost, m_ele, m_roc, sway_Y=1.0):
    years = np.linspace(0, duration, 200)
    costs = []
    
    # 实际电梯运力 (ton/year)
    real_ele_cap = yearly_payload_g_nominal / sway_Y
    # 实际需要的电梯年运量 (如果 m_ele 是总目标)
    # 这里的 m_ele 是 solve 函数算出来的"总量"。
    # 在仿真中，我们假设按分配好的 m_ele 和 m_roc 匀速执行
    step_ele = m_ele / 200
    step_roc = m_roc / 200
    
    cumulative_cost = 0.0
    cumulative_roc_mass = 0.0
    
    avg_base_price = 0
    total_cap = df_bases["Yearly"].sum()
    for _, b in df_bases.iterrows():
        avg_base_price += b["Cost"] * (b["Yearly"]/total_cap)
    
    for _ in years:
        costs.append(cumulative_cost / 1e12)
        cost_step_ele = step_ele * cost_g
        
        if M_TOTAL > 0: progress = cumulative_roc_mass / M_TOTAL
        else: progress = 0
        
        current_discount = SCALE_DISCOUNT * progress
        current_rocket_price = avg_base_price * (1 - current_discount)
        cost_step_roc = step_roc * current_rocket_price
        
        cumulative_cost += (cost_step_ele + cost_step_roc)
        cumulative_roc_mass += step_roc
    
    # 校准终点
    if cumulative_cost > 0: scale_fix = total_cost / cumulative_cost
    else: scale_fix = 1.0
    return years + START_YEAR, [c * scale_fix for c in costs]

# --- 绘制 Method 3 (Hybrid) 的不确定性带 ---
# 选取标称时间 T=150 进行多次仿真
t_target = 150.0
sim_curves_m3 = []
# 为了绘图效率，如果样本太多，只画前 200 个样本的轨迹用于置信区间计算，或者全部计算
# 这里全部计算
for Y_val in sway_samples:
    raw, m_r, m_e, _ = solve_allocation_with_integral_cost(t_target, M_TOTAL, sway_Y=Y_val)
    pres = (T_MAX_NOMINAL - t_target) / (T_MAX_NOMINAL - T_MIN)
    final_c = raw * (1 + 0.5 * pres**2)
    ys, cs = get_curve(t_target, final_c, m_e, m_r, Y_val)
    # 为了绘图，插值到统一时间轴
    common_years = np.linspace(START_YEAR, START_YEAR + 450, 500)
    interp_costs = np.interp(common_years, ys, cs, right=np.nan) # 完成后设为NaN
    sim_curves_m3.append(interp_costs)

sim_arr_m3 = np.array(sim_curves_m3)
mean_m3 = np.nanmean(sim_arr_m3, axis=0)
upper_m3 = np.nanpercentile(sim_arr_m3, 95, axis=0)
lower_m3 = np.nanpercentile(sim_arr_m3, 5, axis=0)



ax.plot(common_years, mean_m3, color=colors['hybrid'], linewidth=2, label='M3: Hybrid Mean')
ax.fill_between(common_years, lower_m3, upper_m3, color=colors['hybrid'], alpha=0.2)

# --- 绘制 Method 1 (Elevator) 的不确定性带 ---
sim_curves_m1 = []
for Y_val in sway_samples:
    real_cap = yearly_payload_g_nominal / Y_val
    dur = M_TOTAL / real_cap
    c_m1 = M_TOTAL * cost_g
    ys, cs = get_curve(dur, c_m1, M_TOTAL, 0, Y_val)
    interp_costs = np.interp(common_years, ys, cs, right=np.nan)
    sim_curves_m1.append(interp_costs)

sim_arr_m1 = np.array(sim_curves_m1)
mean_m1 = np.nanmean(sim_arr_m1, axis=0)
upper_m1 = np.nanpercentile(sim_arr_m1, 95, axis=0)
lower_m1 = np.nanpercentile(sim_arr_m1, 5, axis=0)

ax.plot(common_years, mean_m1, color=colors['elevator'], linewidth=2, label='M1: Elevator Mean')
ax.fill_between(common_years, lower_m1, upper_m1, color=colors['elevator'], alpha=0.2)

# --- 绘制 Method 2 (Rocket) - 稳如泰山 ---
# Method 2 不受影响，画一条实线即可
ys_m2, cs_m2 = get_curve(t_m2, final_cost_m2, 0, M_TOTAL)
ax.plot(ys_m2, cs_m2, color=colors['rocket'], linewidth=2.5, label='M2: Rockets (Deterministic)')


# 装饰
ax.set_title("Robustness Analysis: Cost Trajectories under Uncertainty", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Accumulated Cost (Trillion USD)", fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(START_YEAR, START_YEAR + 450)
ax.legend(loc='upper left')

# Inset Zoom
axins = inset_axes(ax, width="40%", height="40%", loc=1, borderpad=3)
axins.plot(common_years, mean_m3, color=colors['hybrid'])
axins.fill_between(common_years, lower_m3, upper_m3, color=colors['hybrid'], alpha=0.2)
axins.plot(ys_m2, cs_m2, color=colors['rocket'])
axins.set_xlim(2050, 2220)
axins.set_ylim(0, final_cost_m2/1e12 * 1.1)
axins.grid(True, linestyle=':', alpha=0.4)
axins.set_title("Zoom: Uncertainty in Early Phase", fontsize=9)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--")

plt.tight_layout()
plt.savefig("simulation_trajectories_uncertainty.png", dpi=300)
plt.show()

# =============================================================================
# 7. (新增) T=150 成本分布敏感度分析图
# =============================================================================
print(f"正在计算 T=150 的成本分布敏感度...")

# 1. 设定目标参数
target_T = 150.0
costs_at_150 = []

# 2. 计算该时间点的压力惩罚系数 (对所有样本统一，因为 T 固定)
# 注意：T_MAX_NOMINAL 和 T_MIN 是之前全局计算好的变量，直接调用
pressure_150 = max(0, (T_MAX_NOMINAL - target_T) / (T_MAX_NOMINAL - T_MIN))
penalty_factor_150 = 1 + 0.5 * (pressure_150 ** 2)

# 3. 遍历之前的 Monte Carlo 样本重新计算成本
for Y_val in sway_samples:
    # 复用之前的核心求解器
    raw_cost, _, _, _ = solve_allocation_with_integral_cost(target_T, M_TOTAL, sway_Y=Y_val)
    
    # 应用压力惩罚
    final_c = raw_cost * penalty_factor_150
    costs_at_150.append(final_c / 1e12) # 转换为万亿美元

# 4. 绘图：成本分布直方图 + 密度曲线
plt.figure(figsize=(9, 6))

# 绘制直方图 (样本多，bin可以多一点)
counts, bins, patches = plt.hist(costs_at_150, bins=50, color='#7570B3', alpha=0.6, 
                                 edgecolor='black', density=True, label='Frequency')

# 尝试绘制核密度估计曲线 (KDE) 以显示"变化的相对程度"
try:
    from scipy.stats import gaussian_kde
    density = gaussian_kde(costs_at_150)
    xs = np.linspace(min(costs_at_150), max(costs_at_150), 200)
    plt.plot(xs, density(xs), color='red', linestyle='--', linewidth=2, label='Probability Density')
except ImportError:
    pass # 如果没有 scipy 就不画曲线，不影响直方图

# 5. 计算并标注统计指标 (量化相对变化)
mean_cost = np.mean(costs_at_150)
std_cost = np.std(costs_at_150)
cv_cost = (std_cost / mean_cost) * 100 # 变异系数 (Coefficient of Variation)

# 绘制均值线
plt.axvline(mean_cost, color='k', linestyle='dashed', linewidth=1.5, label=f'Mean Cost: {mean_cost:.2f}T')

# 图表装饰
plt.title(f"Cost Uncertainty Distribution at T={target_T} Years\n(Monte Carlo N={N_SAMPLES}, Truncated Normal Input)", fontsize=14, fontweight='bold')
plt.xlabel("Total Project Cost (Trillion USD)", fontsize=12)
plt.ylabel("Probability Density", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.5)

# 在图表中插入文本框，显示相对浮动程度
text_str = (f"Target Time: {target_T} Years\n"
            f"Mean Cost: {mean_cost:.3f}T\n"
            f"Std Dev: {std_cost:.3f}T\n"
            f"Relative Variation (CV): {cv_cost:.2f}%")

plt.text(0.95, 0.95, text_str, transform=plt.gca().transAxes, fontsize=11,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("cost_distribution_at_150.png", dpi=300)
plt.show()

print(f"T={target_T} 的成本分布图已生成: cost_distribution_at_150.png")
print("--- 高数量蒙特卡洛分析完成 ---")