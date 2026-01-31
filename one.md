# 1. 统一符号与决策变量

- 需要送达月球殖民地的“建材净质量”：
  \[
  M = 10^8\ \text{吨}
  \]
- 工期（年）：\(T\)

## 分配变量（这是三种方法对比的核心）

- 方法一（电梯+同步轨道二段火箭）经第 \(i\) 个银河港口送到月球的建材净质量：
  \[
  m_i \ge 0,\quad i=1,2,3
  \]
- 方法二（地面火箭直达）经第 \(j\) 个发射基地送到月球的建材净质量：
  \[
  n_j \ge 0,\quad j\in\mathcal J,\ |\mathcal J|=10
  \]

总任务约束（无论哪种方案最终都要满足）：
\[
\sum_{i=1}^3 m_i + \sum_{j\in\mathcal J} n_j = M
\]

---

# 2. 用火箭方程把“燃料占比差异”写进模型

对任意一次从某起点到月球的火箭段，设：

- 该段所需速度增量：\(\Delta v\)
- 比冲：\(I_{sp}\)
- \(g_0\) 为标准重力加速度
- 结构系数（干质量相对于“有效载荷”的比例）：
  \[
  \alpha = \frac{m_{\text{dry}}}{m_{\text{payload}}}
  \]

火箭方程给出质量比：
\[
R(\Delta v)=\exp\left(\frac{\Delta v}{I_{sp}g_0}\right)
\]

若该段要把**有效载荷** \(m_{\text{payload}}\) 送达（此处有效载荷就是“建材净质量”），则：

- 末质量（到达前、燃料烧完后）：
  \[
  m_f = m_{\text{payload}} + m_{\text{dry}} = (1+\alpha)m_{\text{payload}}
  \]
- 初质量（点火前总质量）：
  \[
  m_0 = R(\Delta v) \cdot m_f = R(\Delta v)(1+\alpha)m_{\text{payload}}
  \]
- 燃料质量：
  \[
  m_{\text{prop}} = m_0 - m_f = \big(R(\Delta v)-1\big)(1+\alpha)m_{\text{payload}}
  \]

**关键解释**：  
\(\Delta v\) 越大 → \(R\) 越大 → 每吨建材需要的燃料越多（燃料占比越大）。这正对应你提到的“同步轨道二段火箭 vs 地面直达火箭，以及不同基地纬度差异”。

---

# 3. 方法一：仅银河港口（电梯 + 同步轨道二段火箭）

## 3.1 同步轨道二段火箭的“质量放大系数”

同步轨道平台（Apex Anchor/近似 GEO）到月球殖民地的二段火箭，设：

- \(\Delta v_G\)：GEO→月球所需速度增量
- \(I_{sp,G}\)：该段比冲
- \(\alpha_G\)：该段结构系数（干质量/建材净质量）

则每 1 吨建材 \(m_i\) 需要在 GEO 处准备的**总初质量**：
\[
m_{0,G}(m_i)= \underbrace{R_G(1+\alpha_G)}_{\kappa_G} \cdot m_i
\]
其中
\[
R_G=\exp\left(\frac{\Delta v_G}{I_{sp,G}g_0}\right), \quad \kappa_G = R_G(1+\alpha_G)
\]

并且燃料质量为：
\[
m_{\text{prop},G}(m_i)=\big(R_G-1\big)(1+\alpha_G)m_i
\]

> **解释**：\(\kappa_G\) 就是你要的“二阶段火箭及其燃料会占据电梯运力”的数学化表达：  
> 电梯要抬上去的不是 \(m_i\)，而是 \(\kappa_G m_i\)。

---

## 3.2 电梯年运力约束（时效约束）

题给每个银河港口年吞吐上限 \(U=179000\) 吨/年。  
我们把它作用在“电梯抬升到同步轨道的总质量”上：

第 \(i\) 个港口在工期 \(T\) 内可抬升的总质量 ≤ \(U T\)：
\[
\kappa_G m_i \le U T, \quad i=1,2,3
\]

因此方法一在时间 \(T\) 内可完成的建材净吞吐（每港口）是 \(UT/\kappa_G\)。

---

## 3.3 成本：电力成本 + 二段火箭（干质量成本+燃料成本）

### (a) 电梯电力成本（你给的“势能/效率×电价”形式）

把每 kg 从地表升到 GEO 的能量按“引力势能差”（更物理）写为：
\[
\Delta \varepsilon
= \mu \left(\frac{1}{R_E}-\frac{1}{r_{GEO}}\right)\quad(\text{J/kg})
\]
\(\mu\) 为地球引力参数，\(R_E\) 地球半径，\(r_{GEO}\) GEO 半径。

若电梯系统总体效率 \(\eta_E\)，电价 \(\pi_E\)（元/kWh），则抬升质量 \(m\)（吨）的电费：
\[
C_{\text{elec}}(m)= \frac{(1000m)\Delta \varepsilon}{\eta_E \cdot 3.6 \times 10^6} \pi_E
\]
（因为 1 吨=1000kg，1kWh=3.6e6J）

对港口 \(i\)，抬升质量是 \(\kappa_G m_i\)，所以：
\[
C_{\text{elec},i} = C_{\text{elec}}(\kappa_G m_i)
\]

> 你也可以用更粗的近似 \(\Delta \varepsilon \approx g h\)；上式只是把它“更严谨”地写成势能差。

### (b) 二段火箭成本（不回收）

设：

- 二段火箭干质量单位成本：\(c_{\text{dry},G}\)（元/吨）
- 燃料单位成本：\(c_{\text{prop},G}\)（元/吨）

干质量为 \(\alpha_G m_i\)，燃料质量为 \((R_G-1)(1+\alpha_G)m_i\)，所以：
\[
C_{\text{rocket},G,i} = c_{\text{dry},G} \cdot \alpha_G m_i + c_{\text{prop},G} \cdot \big(R_G-1\big)(1+\alpha_G)m_i
\]

### 方法一总成本
\[
C^{(1)} = \sum_{i=1}^3 \Big(C_{\text{elec}}(\kappa_G m_i) + C_{\text{rocket},G,i}\Big)
\]

---

# 4. 方法二：仅地面基地火箭直达月球（考虑不同基地燃料占比差异）

## 4.1 不同基地的 \(\Delta v_j\)（纬度/发射方式差异的入口）

设发射基地 \(j\) 纬度为 \(\varphi_j\)。地球自转在赤道的线速度为 \(\omega R_E\)，在纬度 \(\varphi_j\) 的可用“顺行增速”近似为：
\[
v_{\text{rot}}(j) = \omega R_E \cos \varphi_j
\]

把基地差异吸收进 \(\Delta v_j\)：
\[
\Delta v_j = \Delta v_{\text{base}} - v_{\text{rot}}(j) + \delta_j
\]
- \(\Delta v_{\text{base}}\)：从地面到“地月转移并捕获”的参考速度需求（可统一设定）
- \(\delta_j\)：基地特有修正（轨道倾角限制、气象窗口、发射廊道、任务剖面差异等）

> 这就是你说的“基地经纬度不同、发射方式不同 → 燃料占比不同”的数学表达。

质量比：
\[
R_j = \exp\left(\frac{\Delta v_j}{I_{sp,E}g_0}\right)
\]
结构系数 \(\alpha_E\)（地面直达火箭段）则每 1 吨建材净质量需要的点火前总质量：
\[
m_{0,E}(n_j) = R_j (1+\alpha_E) n_j
\]

燃料质量：
\[
m_{\text{prop},E}(n_j) = \big(R_j - 1\big)(1+\alpha_E)n_j
\]

---

## 4.2 时效：基地发射频率约束

设基地 \(j\) 年发射次数上限为 \(\lambda_j\)（次/年）。

为了把“燃料占比会影响单次能送多少建材”也写进时效，我们引入“同型火箭最大点火前质量上限” \(L_E\)（吨/次）。则基地 \(j\) 单次可送的建材净质量上限为：
\[
q_j = \frac{L_E}{R_j(1+\alpha_E)}
\]
于是基地 \(j\) 在 \(T\) 年内可送达：
\[
n_j \le \lambda_j T \cdot \frac{L_E}{R_j(1+\alpha_E)}
\]

> 解释：同样的火箭（同 \(L_E\)），如果基地 \(j\) 的 \(\Delta v_j\) 更大 → \(R_j\) 更大 → \(q_j\) 更小 → 同样频率下的“建材净吞吐”更低。

---

## 4.3 成本：火箭干质量 + 燃料

设：

- 干质量单位成本：\(c_{\text{dry},E}\)（元/吨）
- 燃料单位成本：\(c_{\text{prop},E}\)（元/吨）

则：
\[
C^{(2)} = \sum_{j \in \mathcal J} \Big( c_{\text{dry},E} \cdot \alpha_E n_j + c_{\text{prop},E} \cdot \big(R_j-1\big)(1+\alpha_E)n_j \Big)
\]

（若你还想体现“启用基地的固定成本”，再加 \(+F_j \cdot \mathbb{1}[n_j>0]\) 即可。）

---

# 5. 方法三：两者并行组合（你强调的“并行”在约束里自然体现）

并行意味着在同一个工期 \(T\) 内，电梯路线和直达路线都在送货，所以我们只需同时满足两类容量约束，并满足总量。

## 5.1 组合方案的约束系统
\[
\sum_{i=1}^3 m_i + \sum_{j \in \mathcal J} n_j = M
\]
\[
\kappa_G m_i \le U T, \quad i=1,2,3
\]
\[
n_j \le \lambda_j T \cdot \frac{L_E}{R_j(1+\alpha_E)}, \quad j \in \mathcal J
\]

## 5.2 组合方案总成本
\[
C^{(3)} = C^{(1)} + C^{(2)}
\]

---

# 6. “只考虑成本+时效”的目标函数怎么写（两种标准写法）

这题本质是**双目标优化**：\((C, T)\)。

## 写法 A：加权和（简单好写）
\[
\min \Big( C + \beta T \Big)
\]
\(\beta\) 是“每延长 1 年的时间损失折算成多少钱”的权重。

## 写法 B：\(\varepsilon\)-约束（更符合“给出工期与成本对比”）
给定工期上限 \(T \le T_{\max}\)，最小化成本：
\[
\min C \quad \text{s.t.} \quad T \le T_{\max} \quad \text{以及上面的所有约束}
\]
然后变换 \(T_{\max}\)（比如 50 年、80 年、100 年…），得到成本-工期曲线用于对比三方案。

---

# 7. 三种方法在模型里怎么“切换”

- **方法一**：强制 \(n_j = 0\)
- **方法二**：强制 \(m_i = 0\)
- **方法三**：两者都允许（并行）


---


# 参数与对应值

## 1. **基本参数**

- **建材净质量**：
  \[
  M = 10^8\ \text{吨}
  \]

- **干质量**：
  \[
  m_{\text{dry}} = 400\ \text{吨}
  \]

- **有效载荷（即建材净质量）**：
  \[
  m_{\text{payload}} = 150\ \text{吨}
  \]

- **Δv_G**：GEO → 月球所需速度增量  
  \[
  \Delta v_G = 2.2\ \text{km/s}
  \]

- **比冲**：
  \[
  I_{sp,G} = 380\ \text{s}
  \]

- **μ（地球引力参数）**：
  \[
  \mu = GM \approx 398,600.44\ \frac{\text{km}^3}{\text{s}^2}
  \]

- **地球半径**：
  \[
  R_E = 6,378\ \text{km}
  \]

- **GEO 半径**：
  \[
  r_{\text{GEO}} = 42,164\ \text{km}
  \]

## 2. **电梯系统与电力参数**

- **电梯系统总体效率**：
  \[
  \eta_E = 80\%
  \]

- **电价**：
  \[
  \pi_E = 0.03\ \text{USD/kWh}
  \]

## 3. **火箭成本参数**

- **二段火箭干质量单位成本**：
  \[
  c_{\text{dry},G} \approx 3.1 \times 10^6\ \text{USD/t}
  \]

- **燃料单位成本**：
  \[
  c_{\text{prop},G} \approx 5.0 \times 10^2\ \text{USD/t}
  \]

## 4. **发射基地参数**

### 基地经纬度：

- **Alaska（美国）Pacific Spaceport Complex** – Alaska：  
  \(\varphi_j = 57.43528^\circ \text{N}\)

- **California（美国）Vandenberg Space Force Base**：  
  \(\varphi_j = 34.75133^\circ \text{N}\)

- **Texas（美国）SpaceX Starbase**：  
  \(\varphi_j = 25.99700^\circ \text{N}\)

- **Florida（美国）Cape Canaveral Space Force Station**：  
  \(\varphi_j = 28.48889^\circ \text{N}\)

- **Virginia（美国）Mid-Atlantic Regional Spaceport**：  
  \(\varphi_j = 37.84333^\circ \text{N}\)

- **Kazakhstan Baikonur Cosmodrome**：  
  \(\varphi_j = 45.96500^\circ \text{N}\)

- **French Guiana Guiana Space Centre**：  
  \(\varphi_j = 5.16900^\circ \text{N}\)

- **India Satish Dhawan Space Centre**：  
  \(\varphi_j = 13.72000^\circ \text{N}\)

- **China Taiyuan Satellite Launch Center**：  
  \(\varphi_j = 38.84910^\circ \text{N}\)

- **New Zealand Mahia Peninsula (Rocket Lab LC-1)**：  
  \(\varphi_j = 39.26085^\circ \text{S}\)

### 地球自转线速度：

- **地球自转在赤道的线速度**：
  \[
  v_{\text{rot}} = \omega R_E = 465.1\ \text{m/s}
  \]

## 5. **火箭发射参考参数**

- **Δv_{\text{base}}**：从地面到“地月转移并捕获”的参考速度需求  
  \[
  \Delta v_{\text{base}} = 15.2\ \text{km/s}
  \]

- **比冲**：
  \[
  I_{sp,E} = 350\ \text{s}
  \]

- **同型火箭最大点火前质量上限**：
  \[
  L_E = 5000\ \text{吨/次}
  \]

## 6. **参数假设来源**

1. **电梯系统总体效率 \(\eta_E\)**：
   - 电梯系统由三个环节组成：  
     \(\eta_E = \eta_{\text{transmission}} \times \eta_{\text{conversion}} \times \eta_{\text{mechanical}}\)
   - 假设2050年使用超导缆绳，传输效率为90%，电机转换效率为95%，机械效率为95%，因此：
     \[
     \eta_E = 80\%
     \]

2. **电价 \(\pi_E\)**：
   - 基于赤道地区的太阳能优势，预计到2050年，光伏发电成本将大幅下降，电价为：
     \[
     \pi_E = 0.03\ \text{USD/kWh}
     \]

3. **同型火箭最大点火前质量上限 \(L_E\)**：
   - 参考 SpaceX 超重型火箭，最大有效载荷为 150 吨，起飞质量为 5000 吨，假设该最大点火前质量上限为：
     \[
     L_E = 5000\ \text{吨/次}
     \]
