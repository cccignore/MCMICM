import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import time

# =============================================================================
# 1. 全局配置 (Configuration)
# =============================================================================
class Config:
    # --- 基础物理常数 ---
    G0 = 9.80665
    MU = 398600.44
    RE = 6378.0
    RGEO = 42164.0
    V_ROT_EQUATOR = 0.4651
    
    # --- 结构与推进参数 ---
    ALPHA_G = 0.3               # OTV 结构系数
    ALPHA_E = 0.6               # 地面火箭结构系数
    L_MAX = 6000.0              # 最大起飞质量 (t)
    
    DV_G, ISP_G = 2.2, 460.0    # OTV: GEO -> Moon
    DV_BASE, ISP_E = 15.2, 430.0# GR: Ground -> TLI
    
    # --- 成本基础参数 ---
    C_DRY = 3.1e6               # 干重成本 ($/t)
    C_PROP = 500.0              # 燃料成本 ($/t)
    ETA_E, PI_E = 0.8, 0.03     # 电梯效率与电价
    
    # --- 设施能力 ---
    N_PORTS = 3
    U_PORT = 179000.0           # 电梯物理年吞吐量 (总重)
    LAMBDA_J = 2.0              # 单个基地日发射限制
    
    # --- 仿真环境设置 ---
    T_DAYS = 365
    DEMAND_MEAN = 1000.0
    DEMAND_STD = 100.0
    RECYCLE_RATE = 0.98
    RECYCLE_LAG = 3
    
    # --- 阈值与约束 ---
    I_DEAD = 0.0                # 死亡线
    I_PANIC = 1800.0            # 熔断线
    I_NOM = 2000.0              # 生存底线 (Day 1 目标)
    
    # --- 惩罚系数 ---
    PENALTY_DEATH = 1e18        # 无穷大
    PENALTY_PANIC = 1e12        # 违规重罚
    PENALTY_COLD = 1e13         # 冷启动强制罚款
    
    # --- DP 网格设置 ---
    DP_I_RES = 60               # 库存网格数
    DP_M_RES = 10               # GEO积累网格数
    
    # --- 基地数据 ---
    BASES_DATA = {
        "Alaska (USA)": 57.43528, "Vandenberg (USA)": 34.75133,
        "Starbase (USA)": 25.99700, "Cape Canaveral (USA)": 28.48889,
        "Wallops (USA)": 37.84333, "Baikonur (KAZ)": 45.96500,
        "Kourou (GUF)": 5.16900, "Sriharikota (IND)": 13.72000,
        "Taiyuan (CHN)": 38.84910, "Mahia (NZL)": -39.26085
    }

# =============================================================================
# 2. 物理引擎 (Physics Engine) - 修正了基地差异化计算
# =============================================================================
class PhysicsEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self._init_common_physics()
        self._init_bases_physics()
        
    def _get_mass_and_cost(self, dv, isp, alpha, is_elevator=False):
        """通用火箭方程与成本计算器"""
        r = np.exp((dv * 1000) / (isp * self.cfg.G0))
        kappa = r * (1 + alpha)
        
        m_start = self.cfg.L_MAX
        m_final = m_start / r
        m_prop = m_start - m_final
        m_dry = alpha * m_prop
        
        cost_fixed = m_dry * self.cfg.C_DRY + m_prop * self.cfg.C_PROP
        
        cost_lift_per_ton = 0.0
        if is_elevator:
            delta_epsilon = self.cfg.MU * (1/self.cfg.RE - 1/self.cfg.RGEO) * 1e6
            kwh_per_ton = (1000 * delta_epsilon) / (self.cfg.ETA_E * 3.6e6)
            cost_lift_per_ton = kwh_per_ton * self.cfg.PI_E
            
        return kappa, cost_fixed, cost_lift_per_ton

    def _init_common_physics(self):
        # 1. OTV 参数
        self.kappa_g, self.cost_fixed_otv, _ = self._get_mass_and_cost(
            self.cfg.DV_G, self.cfg.ISP_G, self.cfg.ALPHA_G, is_elevator=False
        )
        self.m_dead = self.cfg.L_MAX * (1 - 1/self.kappa_g)
        self.c_otv_water_max = self.cfg.L_MAX - self.m_dead
        
        # 2. 电梯参数
        _, _, self.cost_se_per_ton_phy = self._get_mass_and_cost(
            0, 1, 0, is_elevator=True 
        )
        self.cap_se_daily_phy = (self.cfg.N_PORTS * self.cfg.U_PORT) / 365.0

    def _init_bases_physics(self):
        self.bases = []
        for name, lat in self.cfg.BASES_DATA.items():
            v_rot = self.cfg.V_ROT_EQUATOR * np.cos(np.radians(lat))
            dv_eff = self.cfg.DV_BASE - v_rot
            
            kappa, cost_fixed, _ = self._get_mass_and_cost(
                dv_eff, self.cfg.ISP_E, self.cfg.ALPHA_E
            )
            payload = self.cfg.L_MAX / kappa
            
            self.bases.append({
                'name': name,
                'payload': payload,
                'cost': cost_fixed,
                'efficiency': cost_fixed / payload 
            })
        
        self.bases.sort(key=lambda x: x['efficiency'])
        
        # 计算全球单日最大物理运力
        self.max_daily_payload = sum([b['payload'] * self.cfg.LAMBDA_J for b in self.bases])
        
        print(f"--- 物理参数初始化 ---")
        print(f"OTV 死重: {self.m_dead:.2f} t, 满载净水: {self.c_otv_water_max:.2f} t")
        print(f"电梯日运力: {self.cap_se_daily_phy:.2f} t")
        print(f"全球地面火箭日最大运力: {self.max_daily_payload:.2f} t (严格限制)")
        print("----------------------")

    def get_otv_water(self, m_accumulated):
        if m_accumulated <= self.m_dead: return 0.0
        return min(self.cfg.L_MAX, m_accumulated) - self.m_dead

    def calculate_gr_mission(self, gap_needed):
        """
        [修正版] 贪心调度算法：严格遵守 Lambda_J 限制
        """
        remaining_gap = gap_needed
        total_cost = 0.0
        total_payload = 0.0
        launch_summary = {} 
        
        # 1. 正常遍历：优先用便宜的
        for base in self.bases:
            if remaining_gap <= 0.001: break
            
            # 严格计算该基地还能发几次
            # 这里的逻辑是单次决策，所以每次进来该基地都有 LAMBDA_J 次机会
            max_count = self.cfg.LAMBDA_J
            
            # 需要多少次？
            needed_count = np.ceil(remaining_gap / base['payload'])
            
            # 取小值：不能超过基地限制
            count = int(min(max_count, needed_count))
            
            if count > 0:
                cost = count * base['cost']
                payload = count * base['payload']
                
                total_cost += cost
                total_payload += payload
                remaining_gap -= payload
                launch_summary[base['name']] = count
        
        # 2. [关键修正] 删除兜底逻辑
        # 如果 remaining_gap > 0，说明全球所有火箭都发了也不够填坑。
        # 此时就让它不够，返回实际能发的最大值。
        # 库存会因此低于预期，这是物理约束的体现。
            
        return total_cost, total_payload, launch_summary

# =============================================================================
# 3. 动态规划求解器 (DPSolver)
# =============================================================================
class DPSolver:
    def __init__(self, cfg, phy):
        self.cfg = cfg
        self.phy = phy
        
        self.I_grid = np.linspace(0, 6000.0, cfg.DP_I_RES)
        self.M_grid = np.linspace(0, cfg.L_MAX + cfg.U_PORT/365.0, cfg.DP_M_RES)
        
        self.V = np.zeros((cfg.T_DAYS + 2, len(self.I_grid), len(self.M_grid)))
        self.Policy = np.zeros((cfg.T_DAYS + 1, len(self.I_grid), len(self.M_grid)), dtype=int)
        
    def _get_penalty(self, I, t):
        if I < self.cfg.I_DEAD: return self.cfg.PENALTY_DEATH
        
        pen = 0.0
        if t == 1 and I < self.cfg.I_NOM:
            # 提高 Day 1 达标惩罚权重，迫使算法尽最大努力
            return self.cfg.PENALTY_COLD
            
        if I < self.cfg.I_PANIC:
            pen += self.cfg.PENALTY_PANIC + 1e7 * (self.cfg.I_PANIC - I)
            
        return pen

    def solve(self):
        print("启动 DP 反向归纳...")
        start_t = time.time()
        c_lift_daily = self.phy.cap_se_daily_phy * self.phy.cost_se_per_ton_phy
        
        for t in range(self.cfg.T_DAYS, 0, -1):
            if t <= self.cfg.RECYCLE_LAG:
                net_demand = self.cfg.DEMAND_MEAN
            else:
                net_demand = self.cfg.DEMAND_MEAN * (1 - self.cfg.RECYCLE_RATE)
            
            v_next_interp = RegularGridInterpolator(
                (self.I_grid, self.M_grid), self.V[t+1], bounds_error=False, fill_value=None
            )
            
            for i_idx, I in enumerate(self.I_grid):
                for m_idx, M in enumerate(self.M_grid):
                    LIMIT = 1e16
                    # Action 0: Wait
                    I_next = I - net_demand
                    M_next = min(self.cfg.L_MAX, M + self.phy.cap_se_daily_phy)
                    pen = self._get_penalty(I_next, t)
                    if pen > LIMIT: v_wait = np.inf
                    else: v_wait = c_lift_daily + pen + v_next_interp((I_next, M_next))
                    
                    # Action 1: OTV
                    v_otv = np.inf
                    if M > self.phy.m_dead:
                        water = self.phy.get_otv_water(M)
                        I_next_otv = I - net_demand + water
                        M_next_otv = 0.0 
                        pen_otv = self._get_penalty(I_next_otv, t)
                        if pen_otv > LIMIT: 
                            v_otv = np.inf
                        else:    
                            v_otv = self.phy.cost_fixed_otv + c_lift_daily + pen_otv + v_next_interp((I_next_otv, M_next_otv))
                            
                    # Action 2: GR
                    gap = max(0, self.cfg.I_NOM - (I - net_demand))
                    
                    # 无论 Gap 是否为 0，都计算如果不发、发一点、发满的代价
                    # 这里简化为：如果有缺口就补，没缺口就发最便宜的保底（为了防止插值空洞）
                    if gap > 0:
                        cost_gr, payload_gr, _ = self.phy.calculate_gr_mission(gap)
                    else:
                        base = self.phy.bases[0]
                        cost_gr = base['cost']
                        payload_gr = base['payload']
                        
                    I_next_gr = I - net_demand + payload_gr
                    M_next_gr = min(self.cfg.L_MAX, M + self.phy.cap_se_daily_phy)
                    
                    pen_gr = self._get_penalty(I_next_gr, t)
                    if pen_gr > LIMIT: v_gr = np.inf
                    else: v_gr = cost_gr + c_lift_daily + pen_gr + v_next_interp((I_next_gr, M_next_gr))
                    
                    costs = [v_wait, v_otv, v_gr]
                    best_act = np.argmin(costs)
                    self.V[t, i_idx, m_idx] = costs[best_act]
                    self.Policy[t, i_idx, m_idx] = best_act
                    
        print(f"DP 完成，耗时 {time.time()-start_t:.2f}s")

    def get_action(self, t, I, M):
        i_idx = (np.abs(self.I_grid - I)).argmin()
        m_idx = (np.abs(self.M_grid - M)).argmin()
        return self.Policy[t, i_idx, m_idx]

# =============================================================================
# 4. 仿真与可视化 (Simulation)
# =============================================================================
def run_simulation(cfg, phy, solver):
    print("\n开始正向仿真验证...")
    I, M = 0.0, 0.0
    
    # 【修改点】：初始化时加入 Day 0 的数据 (0成本, 0库存)
    # 这样画图时就会有一条从 (0,0) 到 (1, Cost) 的斜线，直观显示第一天的花销
    history = {
        'day': [0], 
        'I': [0.0], 
        'M': [0.0], 
        'action': ['Start'], 
        'cost': [0.0], 
        'launch_info': ['Init']
    }
    
    # 无回收历史
    demand_hist = [0.0] * cfg.RECYCLE_LAG
    cum_cost = 0.0
    
    for t in range(1, cfg.T_DAYS + 1):
        D = np.random.normal(cfg.DEMAND_MEAN, cfg.DEMAND_STD)
        recycle = demand_hist.pop(0) * cfg.RECYCLE_RATE
        demand_hist.append(D)
        net_demand = max(0, D - recycle)
        
        act = solver.get_action(t, I, M)
        
        cost_step = 0.0
        info = ""
        action_name = "Wait"
        
        real_lifted_mass = min(cfg.L_MAX, M + phy.cap_se_daily_phy) - M
        c_lift = real_lifted_mass * phy.cost_se_per_ton_phy
        cost_step += c_lift
        M_new = min(cfg.L_MAX, M + phy.cap_se_daily_phy)
        I_new = I - net_demand
        
        if act == 1: # OTV
            if M > phy.m_dead:
                cost_step += phy.cost_fixed_otv
                water = phy.get_otv_water(M)
                I_new += water
                M_new = 0.0
                action_name = "OTV"
                info = f"Payload={water:.1f}t"
        
        elif act == 2: # GR
            action_name = "GR"
            gap = max(0, cfg.I_NOM - I_new)
            if gap <= 0: gap = 1.0 
            
            c_gr, p_gr, launch_dict = phy.calculate_gr_mission(gap)
            cost_step += c_gr
            I_new += p_gr
            info_str = ", ".join([f"{k}x{v}" for k,v in launch_dict.items()])
            info = f"Payload={p_gr:.1f}t ({info_str})"
            
        I, M = I_new, M_new
        cum_cost += cost_step
        
        history['day'].append(t)
        history['I'].append(I)
        history['M'].append(M)
        history['action'].append(action_name)
        history['cost'].append(cum_cost/1e9) # 单位 Billion
        history['launch_info'].append(info)
        
        if t == 1 or act != 0:
            print(f"Day {t:3d}: I={I:.1f}, M={M:.1f} | {action_name} | {info} | Cost +${cost_step/1e6:.1f}M")
            
    return pd.DataFrame(history)

if __name__ == "__main__":
    # 实例化 (保持原逻辑)
    cfg = Config()
    phy = PhysicsEngine(cfg)
    solver = DPSolver(cfg, phy)
    solver.solve()
    df = run_simulation(cfg, phy, solver)
    
    # --- 绘图部分的微调 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # 1. 库存图
    ax1.plot(df['day'], df['I'], label='Inventory', color='tab:blue', lw=2)
    ax1.axhline(cfg.I_NOM, color='green', ls='--', label='Survival (2000t)')
    ax1.axhline(cfg.I_PANIC, color='red', ls=':', label='Panic (1800t)')
    
    # 过滤掉 Day 0 的 Start 动作，只画实际发射
    otv = df[df['action'] == "OTV"]
    gr = df[df['action'] == "GR"]
    ax1.scatter(otv['day'], otv['I'], c='purple', marker='^', s=60, label='OTV Launch', zorder=5)
    ax1.scatter(gr['day'], gr['I'], c='orange', marker='x', s=80, label='Ground Rocket', zorder=5)
    
    ax1.set_ylabel("Water Inventory (t)")
    ax1.set_title("Dual-Track Supply Chain Simulation (With t=0 Init)")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0) # 强制 X 轴从 0 开始
    
    # 2. 成本图
    # 这里会自动把 (0, 0) 到 (1, 216) 连起来，显示出陡峭的爬升
    ax2.plot(df['day'], df['cost'], color='tab:green', lw=2)
    
    # 也可以标记出 Day 1 的成本点
    cost_day1 = df.loc[df['day']==1, 'cost'].values[0]
    ax2.scatter([1], [cost_day1], color='red', s=50, zorder=5)
    ax2.text(1.5, cost_day1, f"Day 1: ${cost_day1:.1f}B", va='bottom', fontsize=9, color='darkgreen')

    ax2.set_ylabel("Cumulative Cost (Billion $)")
    ax2.set_xlabel("Day")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0) # 强制 Y 轴从 0 开始
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n最终总成本: ${df['cost'].iloc[-1]:.2f} Billion")