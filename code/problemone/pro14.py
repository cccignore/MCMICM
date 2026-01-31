import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
# 2. 核心算法: 动态迭代求解器 (积分修正版)
# =============================================================================
def solve_allocation_with_integral_cost(time_target, total_mass):
    final_marginal_discount = 0.0 
    for i in range(10):
        # 使用平均折扣进行决策估算
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
                # 使用积分平均折扣计算总成本
                cost_iter += can_take * b["Cost"] * (1 - avg_discount)
                mass_roc += can_take
                needed_rocket -= can_take
        
        new_marginal_discount = SCALE_DISCOUNT * (mass_roc / total_mass)
        if abs(new_marginal_discount - final_marginal_discount) < 0.0001:
            return cost_iter, mass_roc, mass_ele, new_marginal_discount
        final_marginal_discount = new_marginal_discount
    return cost_iter, mass_roc, mass_ele, final_marginal_discount

# --- 计算三个方案的"终点参数" ---

# Method 1 (Elevator)
t_m1 = T_MAX
m_ele_m1, m_roc_m1 = M_TOTAL, 0
cost_m1 = (M_TOTAL * cost_g) / 1e12 

# Method 2 (Rocket)
t_m2 = M_TOTAL / total_yearly_rockets
# 纯火箭: 平均折扣 = 最大折扣的一半
c_m2_raw = 0
avg_discount_m2 = SCALE_DISCOUNT * 0.5
for _, b in df_bases.iterrows():
    c_m2_raw += b["Yearly"] * t_m2 * b["Cost"]
c_m2_raw *= (1 - avg_discount_m2)
pressure_m2 = (T_MAX - t_m2) / (T_MAX - T_MIN)
final_cost_m2 = c_m2_raw * (1 + 0.5 * pressure_m2**2)

# Method 3 (Hybrid)
t_m3 = 150.0
c_m3_raw, m_roc_m3, m_ele_m3, d_marginal_m3 = solve_allocation_with_integral_cost(t_m3, M_TOTAL)
pressure_m3 = (T_MAX - t_m3) / (T_MAX - T_MIN)
final_cost_m3 = c_m3_raw * (1 + 0.5 * pressure_m3**2)

# =============================================================================
# 3. 真实仿真: 逐年累加计算器
# =============================================================================
def simulate_real_accumulation(duration, target_total_cost, m_ele_total, m_roc_total, label, color, ax):
    """
    仿真并绘制到指定的 ax 上
    """
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
            
        # 这里的单价是当前的边际单价，即 SCALE_DISCOUNT * progress
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
    
    # 绘图 (实线)
    ax.plot(years + START_YEAR, final_costs, color=color, linestyle='-', linewidth=2.5, label=label)
    
    # 终点标注
    end_t = START_YEAR + duration
    end_c = final_costs[-1]
    ax.scatter(end_t, end_c, color=color, s=80, zorder=5, edgecolor='white', linewidth=1.5)
    
    # 根据年份决定标注位置，避免遮挡
    offset = (10, 5) if end_t < 2200 else (-40, 10)
    ax.annotate(f'{int(end_t)}\n${end_c:.1f}T', (end_t, end_c), 
                 xytext=offset, textcoords="offset points", 
                 color=color, fontweight='bold', fontsize=9, va='center')

# =============================================================================
# 4. 绘图 (使用非线性比例尺/双坐标轴)
# =============================================================================

# 科研低饱和配色 (Low Saturation Colors)
# 1. 砖红色 (Muted Red) -> 火箭
# 2. 鼠尾草绿 (Sage Green) -> 电梯
# 3. 深灰蓝 (Slate Blue) -> 混合
colors = {
    'rocket': '#D95F02',   # 砖红
    'elevator': '#1B9E77', # 蓝绿/深绿
    'hybrid': '#7570B3'    # 蓝紫
}

fig, ax = plt.subplots(figsize=(12, 7))

# --- 关键修改: 使用线性轴，但手动设置 limit 来聚焦前半段 ---
# 既然大部分有趣的曲线都在前150年，我们让 X 轴的范围显示完整，
# 但不需要非线性变换，直接画图即可。
# 如果想突出前半段，我们可以在代码逻辑上不做手脚，而是通过图表布局让它看起来舒服。
# 这里采用标准的线性轴，因为时间流逝是均匀的，扭曲它可能会误导读者认为"时间过得快慢不一"。
# 我们通过图例和网格来辅助阅读。

# 1. 绿线: 纯电梯
simulate_real_accumulation(t_m1, cost_m1 * 1e12, M_TOTAL, 0, 
                           'Method 1: Elevator Only', colors['elevator'], ax)

# 2. 红线: 纯火箭
simulate_real_accumulation(t_m2, final_cost_m2, 0, M_TOTAL, 
                           'Method 2: Rockets Only', colors['rocket'], ax)

# 3. 蓝线: 混合
simulate_real_accumulation(t_m3, final_cost_m3, m_ele_m3, m_roc_m3, 
                           'Method 3: Hybrid Strategy', colors['hybrid'], ax)

# --- 装饰与优化 ---
ax.set_title("Cumulative Cost Trajectories: Learning Curve Impact", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Accumulated Cost (Trillion USD)", fontsize=12)

# 设置网格
ax.grid(True, which='major', linestyle='-', alpha=0.3, color='gray')
ax.grid(True, which='minor', linestyle=':', alpha=0.15, color='gray')
ax.minorticks_on()

# 设置 X 轴范围 (完整显示，但重点在于标注清晰)
ax.set_xlim(START_YEAR, START_YEAR + 450)
ax.set_ylim(0, max(final_cost_m2, final_cost_m3, cost_m1*1e12)/1e12 * 1.1)

# 自定义图例 (去掉边框，放在左上角空白处)
ax.legend(loc='upper left', frameon=False, fontsize=11)

# --- 添加"聚焦区域"背景色 (可选) ---
# 给 2050 - 2200 添加非常淡的背景，暗示这是"高强度建设期"
ax.axvspan(2050, 2200, color='gray', alpha=0.05)
ax.text(2125, ax.get_ylim()[1]*0.05, "Active Phase", ha='center', color='gray', fontsize=10, alpha=0.6)

# --- 插入子图 (Inset Zoom) ---
# 为了解决你说的"弧度不明显"的问题，我们在图的右下角放一个放大镜子图，
# 专门展示 2050-2200 这一段的细节。
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 位置: 宽度40%, 高度40%, 放在右下角 (loc=4)
axins = inset_axes(ax, width="40%", height="40%", loc=1, borderpad=3)

# 在子图上再画一遍
simulate_real_accumulation(t_m1, cost_m1 * 1e12, M_TOTAL, 0, '', colors['elevator'], axins)
simulate_real_accumulation(t_m2, final_cost_m2, 0, M_TOTAL, '', colors['rocket'], axins)
simulate_real_accumulation(t_m3, final_cost_m3, m_ele_m3, m_roc_m3, '', colors['hybrid'], axins)

# 设置子图范围: 只看 2050 到 2220 年
axins.set_xlim(2050, 2220)
# Y轴范围自动适应这一段的最大值 (主要是红线的终点)
axins.set_ylim(0, final_cost_m2/1e12 * 1.1)

# 子图装饰
axins.grid(True, linestyle=':', alpha=0.4)
axins.set_title("Zoom: 2050-2220 (Curvature Detail)", fontsize=9)
axins.tick_params(labelsize=8)

# 添加连接线，指出子图是哪里的放大
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="0.5", linestyle="--")

plt.tight_layout()
plt.savefig("final_scientific_plot.png", dpi=300) # 高分辨率保存
plt.show()

print("--- 绘图完成 ---")
print("1. 配色已更新为科研低饱和色系。")
print("2. 所有线条改为实线。")
print("3. 添加了局部放大图 (Inset Plot) 来专门展示前150年的曲线弧度，解决了比例尺导致的压缩问题。")