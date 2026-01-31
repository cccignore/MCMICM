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
# 2. LHS 采样与截断正态分布生成器 (新增模块)
# =============================================================================
def generate_lhs_samples(n_samples=50, mean=1.05, std=0.05, lower_bound=1.0):
    """
    生成符合截断正态分布的 LHS 样本 Y (摇晃系数)
    Y >= 1.0, 位于分母, Y越大运力越低
    """
    # 1. 生成 [0, 1] 之间的均匀分布 LHS 样本
    # 将概率空间分成 n_samples 份
    intervals = np.linspace(0, 1, n_samples + 1)
    # 在每个区间内随机取一个点
    points = np.random.uniform(intervals[:-1], intervals[1:], n_samples)
    # 打乱顺序 (虽然一维不需要打乱，但为了严谨性)
    np.random.shuffle(points)
    
    # 2. 使用逆变换采样 (Inverse Transform Sampling) 将均匀分布映射到正态分布
    # 这里我们手动实现一个简单的映射逻辑，或者直接利用截断逻辑
    # 为了简单且不引入 scipy，我们使用 Accept-Reject 策略配合 LHS 的分层思想
    # 但为了保证 LHS 的均匀性，最佳方法是直接映射：
    
    # 既然不能用 scipy.stats.norm.ppf，我们用 numpy 近似或直接采样后排序匹配
    # 方法：生成大量正态分布随机数，截断，然后取分位数匹配 LHS 的 points
    
    pool_size = n_samples * 100
    raw_pool = np.random.normal(mean, std, pool_size)
    valid_pool = raw_pool[raw_pool >= lower_bound] # 截断
    
    # 如果有效池不够，继续生成 (极小概率)
    while len(valid_pool) < n_samples:
        more = np.random.normal(mean, std, pool_size)
        valid_pool = np.concatenate([valid_pool, more[more >= lower_bound]])
    
    # 排序有效池
    valid_pool.sort()
    
    # 根据 LHS 的 points (百分比位置) 选取对应的值
    # 比如 points=[0.1, 0.5, 0.9], 我们就取 valid_pool 中第 10%, 50%, 90% 位置的数
    indices = (points * len(valid_pool)).astype(int)
    lhs_samples = valid_pool[indices]
    
    return lhs_samples

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
# 4. 执行 LHS 采样与不确定性分析
# =============================================================================
print("执行 LHS 采样与不确定性分析...")
N_SAMPLES = 50 # 样本数
sway_samples = generate_lhs_samples(n_samples=N_SAMPLES, mean=1.05, std=0.1, lower_bound=1.0)

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

# 2. 循环计算 LHS 样本 (Method 1 & 3)
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

plt.title("Pareto Frontier under Elevator Sway Uncertainty (LHS Sampling)", fontsize=14)
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

print("--- 不确定性分析完成 ---")