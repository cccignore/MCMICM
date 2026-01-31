import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# 1. 基础参数
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
    base_list.append({"Cost": c, "Yearly": max_y})
df_bases = pd.DataFrame(base_list).sort_values("Cost")
total_yearly_rockets = df_bases["Yearly"].sum()
T_MIN = M_TOTAL / (yearly_payload_g + total_yearly_rockets)
T_MAX = M_TOTAL / yearly_payload_g

# =============================================================================
# 2. 核心算法: 动态迭代求解器
# =============================================================================
def solve_allocation_with_dynamic_scale(time_target, total_mass):
    current_discount = 0.0 
    for i in range(10):
        effective_base_costs = df_bases["Cost"] * (1 - current_discount)
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
                cost_iter += can_take * b["Cost"] * (1 - current_discount)
                mass_roc += can_take
                needed_rocket -= can_take
        new_discount = SCALE_DISCOUNT * (mass_roc / total_mass)
        if abs(new_discount - current_discount) < 0.0001:
            return cost_iter, mass_roc, mass_ele, new_discount
        current_discount = new_discount
    return cost_iter, mass_roc, mass_ele, current_discount

# --- 计算三个方案的"终点参数" (总时间, 总运量分配) ---

# Method 1 (Elevator)
t_m1 = T_MAX
m_ele_m1, m_roc_m1 = M_TOTAL, 0
# 【修复点】补上了 cost_m1 的定义
cost_m1 = (M_TOTAL * cost_g) / 1e12 

# Method 2 (Rocket)
t_m2 = M_TOTAL / total_yearly_rockets
c_m2_raw, m_roc_m2, m_ele_m2, d_m2 = solve_allocation_with_dynamic_scale(t_m2, M_TOTAL)
pressure_m2 = (T_MAX - t_m2) / (T_MAX - T_MIN)
final_cost_m2 = c_m2_raw * (1 + 0.5 * pressure_m2**2) # 这是包含罚款的最终总价 (Raw USD)

# Method 3 (Hybrid)
t_m3 = 150.0
c_m3_raw, m_roc_m3, m_ele_m3, d_m3 = solve_allocation_with_dynamic_scale(t_m3, M_TOTAL)
pressure_m3 = (T_MAX - t_m3) / (T_MAX - T_MIN)
final_cost_m3 = c_m3_raw * (1 + 0.5 * pressure_m3**2) # Raw USD

# =============================================================================
# 3. 真实仿真: 逐年累加计算器
# =============================================================================
def simulate_real_accumulation(duration, target_total_cost, m_ele_total, m_roc_total, label, color, style):
    """
    不使用任何幂函数。完全通过模拟每一年的运量和当年的单价来累加成本。
    target_total_cost: 最终目标总成本 (Raw USD)
    """
    years = np.linspace(0, duration, 1000) # 模拟200个时间步
    costs = []
    
    # 每一时间步的平均运量
    step_ele = m_ele_total / 1000
    step_roc = m_roc_total / 1000
    
    cumulative_cost = 0.0
    cumulative_roc_mass = 0.0
    
    # 预先计算火箭的基础平均单价 (不打折时)
    avg_base_price = 0
    total_cap = df_bases["Yearly"].sum()
    for _, b in df_bases.iterrows():
        avg_base_price += b["Cost"] * (b["Yearly"]/total_cap)
        
    for _ in years:
        costs.append(cumulative_cost / 1e12) # 转兆美元用于绘图
        
        # 1. 电梯成本
        cost_step_ele = step_ele * cost_g
        
        # 2. 火箭成本 (动态单价)
        if M_TOTAL > 0:
            progress = cumulative_roc_mass / M_TOTAL
        else:
            progress = 0
            
        current_discount = SCALE_DISCOUNT * progress
        current_rocket_price = avg_base_price * (1 - current_discount)
        cost_step_roc = step_roc * current_rocket_price
        
        # 3. 累加
        cumulative_cost += (cost_step_ele + cost_step_roc)
        cumulative_roc_mass += step_roc
        
    # --- 归一化修正 ---
    # 将模拟的累计值对齐到包含罚款的 target_total_cost
    if cumulative_cost > 0:
        scale_fix = target_total_cost / cumulative_cost
    else:
        scale_fix = 1.0
        
    final_costs = [c * scale_fix for c in costs]
    
    # 绘图
    plt.plot(years + START_YEAR, final_costs, color=color, linestyle=style, linewidth=3, label=label)
    
    # 终点标注
    end_t = START_YEAR + duration
    end_c = final_costs[-1]
    plt.scatter(end_t, end_c, color=color, s=120, zorder=5)
    plt.annotate(f'Year {int(end_t)}\n${end_c:.1f}T', (end_t, end_c), 
                 xytext=(10, 0), textcoords="offset points", 
                 color=color, fontweight='bold', va='center')

# =============================================================================
# 4. 绘图
# =============================================================================
plt.figure(figsize=(12, 8))

# 1. 绿线: 纯电梯 (使用 cost_m1*1e12 还原为 Raw USD)
simulate_real_accumulation(t_m1, cost_m1 * 1e12, M_TOTAL, 0, 
                           'Method 1: Elevator (Linear)', 'green', '-')

# 2. 红线: 纯火箭
simulate_real_accumulation(t_m2, final_cost_m2, 0, M_TOTAL, 
                           'Method 2: Rockets (Real Learning Curve)', 'red', '--')

# 3. 蓝线: 混合
simulate_real_accumulation(t_m3, final_cost_m3, m_ele_m3, m_roc_m3, 
                           'Method 3: Hybrid', 'blue', '-.')

plt.title("True Simulation: Cost Accumulation based on Cumulative Volume", fontsize=15)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Accumulated Cost (Trillion USD)", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left', fontsize=11)
plt.xlim(START_YEAR, START_YEAR + 450)

plt.tight_layout()
plt.savefig("true_simulation_curve_fixed.png")
plt.show()

print("--- 仿真完成 ---")
print(f"纯电梯总价: {cost_m1:.2f} T")
print(f"纯火箭总价: {final_cost_m2/1e12:.2f} T")
print(f"混合方案总价: {final_cost_m3/1e12:.2f} T")