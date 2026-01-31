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
SCALE_DISCOUNT = 0.2  # 最终最大折扣 20%
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
# 2. 核心算法修正: 基于积分均值的动态求解器
# =============================================================================
def solve_allocation_with_integral_cost(time_target, total_mass):
    """
    修正逻辑：
    不再假设所有火箭都按'最终最低价'结算。
    而是承认：第一枚是原价，最后一枚是最低价。
    因此，计算总成本时，使用的应当是【平均折扣】，即最终折扣的一半。
    """
    final_marginal_discount = 0.0 
    
    for i in range(10): # 迭代收敛
        # 1. 计算"平均"火箭成本 (用于估算总价)
        # 如果边际成本线性下降，平均折扣 = 最终边际折扣 * 0.5
        avg_discount = final_marginal_discount * 0.5
        
        effective_base_costs = df_bases["Cost"] * (1 - avg_discount)
        avg_rocket_cost = effective_base_costs.mean()
        
        # 2. 贪心分配 (基于平均成本做决策)
        rem = total_mass
        cost_iter = 0
        mass_ele = 0
        mass_roc = 0
        
        # 策略判断
        if avg_rocket_cost < cost_g:
            # 极端情况：平均下来的火箭比电梯便宜
            can_take_roc = min(rem, total_yearly_rockets * time_target)
            # 这里的成本计算也要用平均折扣
            cost_iter += can_take_roc * avg_rocket_cost 
            mass_roc += can_take_roc
            rem -= can_take_roc
            cost_iter += rem * cost_g
            mass_ele += rem
        else:
            # 正常情况
            can_take_ele = min(rem, yearly_payload_g * time_target)
            cost_iter += can_take_ele * cost_g
            mass_ele += can_take_ele
            rem -= can_take_ele
            
            needed_rocket = rem
            for idx, b in df_bases.iterrows():
                if needed_rocket <= 0: break
                can_take = min(needed_rocket, b["Yearly"] * time_target)
                # 【核心修正】: 
                # 计算这批火箭的总价时，使用 (1 - avg_discount)
                # 这相当于对线性下降的成本曲线进行了积分
                cost_iter += can_take * b["Cost"] * (1 - avg_discount)
                mass_roc += can_take
                needed_rocket -= can_take
        
        # 3. 更新"最终边际折扣"
        # 赖特定律/规模效应是基于总量的，这里的 SCALE_DISCOUNT 指的是运完所有量后的最大边际折扣
        new_marginal_discount = SCALE_DISCOUNT * (mass_roc / total_mass)
        
        if abs(new_marginal_discount - final_marginal_discount) < 0.0001:
            # 返回: 总成本, 火箭量, 电梯量, 平均折扣(用于后续显示)
            return cost_iter, mass_roc, mass_ele, new_marginal_discount * 0.5
        
        final_marginal_discount = new_marginal_discount
        
    return cost_iter, mass_roc, mass_ele, final_marginal_discount * 0.5

# =============================================================================
# 3. 生成帕累托数据
# =============================================================================
t_axis = np.linspace(T_MIN, T_MAX, 100)
pareto_data = []

for T in t_axis:
    # 调用积分修正后的求解器
    raw_cost, m_roc, m_ele, avg_discount = solve_allocation_with_integral_cost(T, M_TOTAL)
    
    pressure = (T_MAX - T) / (T_MAX - T_MIN)
    final_cost = raw_cost * (1 + 0.5 * (pressure ** 2))
    
    pareto_data.append({
        "Time": T, 
        "Cost": final_cost / 1e12
    })

df_p = pd.DataFrame(pareto_data)

# =============================================================================
# 4. 计算基准点 (同样应用积分逻辑)
# =============================================================================

# --- 绿点: 纯电梯 ---
t_m1 = T_MAX
cost_m1 = (M_TOTAL * cost_g) / 1e12

# --- 红点: 纯火箭 ---
t_m2 = M_TOTAL / total_yearly_rockets
# 纯火箭意味着 mass_roc = M_TOTAL，所以最终边际折扣 = SCALE_DISCOUNT
# 平均折扣 = SCALE_DISCOUNT * 0.5
avg_discount_m2 = SCALE_DISCOUNT * 0.5

raw_cost_m2 = 0
for _, b in df_bases.iterrows():
    raw_cost_m2 += b["Yearly"] * t_m2 * b["Cost"]
# 应用平均折扣 (积分效果)
raw_cost_m2 *= (1 - avg_discount_m2)

pressure_m2 = (T_MAX - t_m2) / (T_MAX - T_MIN)
final_cost_m2 = raw_cost_m2 * (1 + 0.5 * (pressure_m2 ** 2))
cost_m2 = final_cost_m2 / 1e12

# =============================================================================
# 5. 绘图
# =============================================================================
plt.figure(figsize=(12, 7))

plt.plot(df_p["Time"], df_p["Cost"], 'b-', linewidth=3, label='Method 3: Hybrid Pareto (Integral Cost)')
plt.fill_between(df_p["Time"], df_p["Cost"], color='blue', alpha=0.1)

plt.scatter(t_m1, cost_m1, color='green', s=120, zorder=5, label='Method 1: Elevator Only')
plt.annotate(f'M1\n({t_m1:.1f}y, ${cost_m1:.1f}T)', (t_m1, cost_m1), 
             xytext=(-40, 10), textcoords='offset points', color='green', fontweight='bold')

plt.scatter(t_m2, cost_m2, color='red', s=120, zorder=5, label='Method 2: Rockets Only')
plt.annotate(f'M2\n({t_m2:.1f}y, ${cost_m2:.1f}T)', (t_m2, cost_m2), 
             xytext=(20, 10), textcoords='offset points', color='red', fontweight='bold')

closest_idx = (np.abs(df_p['Time'] - t_m2)).argmin()
hybrid_cost_at_m2 = df_p.iloc[closest_idx]['Cost']
plt.vlines(t_m2, hybrid_cost_at_m2, cost_m2, colors='red', linestyles=':', alpha=0.5)
plt.text(t_m2, (cost_m2 + hybrid_cost_at_m2)/2, "  ", color='red', fontsize=9, rotation=90, va='center')

plt.title("Advanced Pareto Frontier: Integral Cost Calculation (True Learning Curve)", fontsize=14)
plt.xlabel("Completion Time (Years)", fontsize=12)
plt.ylabel("Total Project Cost (Trillion USD)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig("pareto_integral_corrected.png")
plt.show()

print(f"--- 修正后的验证 ---")
print(f"纯火箭方案成本: {cost_m2:.2f} T")
print(f"说明: 这个成本是基于'每一枚火箭价格逐渐降低'积分计算出来的，")
print(f"而不是简单粗暴地用最低价乘以总数。这更加真实。")