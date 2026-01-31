import math
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linprog
import matplotlib.pyplot as plt
# ============================================================
# 0) Inputs
# ============================================================
M = 1e8  # total net material to deliver (tons)
g0 = 9.80665  # m/s^2

# Rocket structural coefficient (same for stage2 and ground-direct)
m_dry = 150.0          # tons
m_payload_ref = 150.0  # tons
alpha = m_dry / m_payload_ref  # alpha_G = alpha_E  (=1)

# Rocket costs (same for stage2 and ground-direct)
c_dry = 3.1e6  # USD/ton (dry)
c_prop = 5.0e2 # USD/ton (propellant)

# ---------------- Method 1: Elevator + GEO->Moon stage ----------------
dv_G = 2.2e3    # m/s
Isp_G = 480.0   # s

U = 179000.0    # tons/year per port (lifted mass)
num_ports = 3

mu = 398600.44 * 1e9  # km^3/s^2 -> m^3/s^2
R_E = 6378e3          # m
r_GEO = 42164e3       # m
eta_E = 0.8
pi_E = 0.03           # USD/kWh

# NEW: Method 1 per-launch payload cap (net tons) + launch cadence per port
L_G = 150.0     # tons net payload per Method1 rocket (strict cap)
nu_G = 712.0    # launches/year/port for Method1

# ---------------- Method 2: Ground direct rocket ----------------
dv_base = 15.2e3      # m/s
vrot_equator = 465.1  # m/s
Isp_E = 450.0         # s
L_E = 6000.0          # tons/launch (interpreted as liftoff mass limit m0)

lambda_j = 712.0      # launches/year/base

bases = [
    ("Alaska PSC", 57.43528),
    ("Vandenberg CA", 34.75133),
    ("Starbase TX", 25.99700),
    ("Cape Canaveral FL", 28.48889),
    ("MARS Virginia", 37.84333),
    ("Baikonur", 45.96500),
    ("Guiana Space Centre", 5.16900),
    ("Satish Dhawan India", 13.72000),
    ("Taiyuan China", 38.84910),
    ("Mahia NZ", -39.26085),
]

# ============================================================
# 1) Method 1 intermediates
# ============================================================
R_G = math.exp(dv_G / (Isp_G * g0))
kappa_G = R_G * (1 + alpha)

fuel_per_t_net_G = (R_G - 1) * (1 + alpha)  # tons fuel per ton net material
dry_per_t_net = alpha                       # tons dry per ton net material
m0_per_t_net_G = kappa_G                    # tons at ignition per ton net

# Elevator energy & cost
delta_eps = mu * (1 / R_E - 1 / r_GEO)      # J/kg
kwh_per_kg_ideal = delta_eps / 3.6e6
kwh_per_kg_in = kwh_per_kg_ideal / eta_E
elec_cost_per_t_lifted = kwh_per_kg_in * 1000 * pi_E
elec_cost_per_t_net_method1 = elec_cost_per_t_lifted * kappa_G

# Rocket cost per ton net (GEO->Moon stage)
rocket_cost_per_t_net_G = c_dry * dry_per_t_net + c_prop * fuel_per_t_net_G

# Total method1 cost per ton net
method1_cost_per_t_net = elec_cost_per_t_net_method1 + rocket_cost_per_t_net_G

# Method1 throughput (net tons/year) - elevator-only
net_per_port_per_year_elevator = U / kappa_G
net_all_ports_per_year_elevator = num_ports * net_per_port_per_year_elevator

# Method1 throughput (net tons/year) - launch cadence & payload cap
net_per_port_per_year_launchcap = nu_G * L_G
net_all_ports_per_year_launchcap = num_ports * net_per_port_per_year_launchcap

# Effective per-port net throughput is min(elevator, launchcap)
net_per_port_per_year_effective = min(net_per_port_per_year_elevator, net_per_port_per_year_launchcap)
net_all_ports_per_year_effective = num_ports * net_per_port_per_year_effective

# ============================================================
# 2) Method 2 intermediates per base
# ============================================================
rows = []
for name, lat_deg in bases:
    lat_rad = math.radians(lat_deg)
    vrot = vrot_equator * math.cos(lat_rad)  # m/s
    dv_j = dv_base - vrot                    # m/s
    R_j = math.exp(dv_j / (Isp_E * g0))

    # With L_E as liftoff mass m0 limit:
    payload_per_launch = L_E / (R_j * (1 + alpha))  # tons net/launch
    payload_per_year = lambda_j * payload_per_launch

    fuel_per_t_net_j = (R_j - 1) * (1 + alpha)
    cost_per_t_net_j = c_dry * alpha + c_prop * fuel_per_t_net_j

    rows.append({
        "base": name,
        "lat_deg": lat_deg,
        "v_rot_mps": vrot,
        "dv_j_mps": dv_j,
        "R_j": R_j,
        "payload_per_launch_t_net": payload_per_launch,
        "payload_per_year_t_net": payload_per_year,
        "fuel_per_t_net_t": fuel_per_t_net_j,
        "cost_per_t_net_USD": cost_per_t_net_j,
    })

df_bases = pd.DataFrame(rows)
net_method2_per_year = df_bases["payload_per_year_t_net"].sum()

# ============================================================
# 3) Export intermediates to Excel
# ============================================================
out_path = Path("intermediate_calcs.xlsx")

inputs_df = pd.DataFrame([
    ["M (tons)", M],
    ["g0 (m/s^2)", g0],
    ["alpha (=m_dry/m_payload_ref)", alpha],
    ["c_dry (USD/t)", c_dry],
    ["c_prop (USD/t)", c_prop],

    ["dv_G (m/s)", dv_G],
    ["Isp_G (s)", Isp_G],
    ["U per port (t/yr lifted)", U],
    ["ports", num_ports],
    ["L_G Method1 payload cap (t net/launch)", L_G],
    ["nu_G Method1 launches/yr/port", nu_G],

    ["mu (m^3/s^2)", mu],
    ["R_E (m)", R_E],
    ["r_GEO (m)", r_GEO],
    ["eta_E", eta_E],
    ["pi_E (USD/kWh)", pi_E],

    ["dv_base (m/s)", dv_base],
    ["vrot_equator (m/s)", vrot_equator],
    ["Isp_E (s)", Isp_E],
    ["L_E (t/launch liftoff m0)", L_E],
    ["lambda_j (launches/yr/base)", lambda_j],
], columns=["parameter", "value"])

method1_df = pd.DataFrame([
    ["R_G", R_G],
    ["kappa_G", kappa_G],
    ["fuel_per_t_net_G (t/t)", fuel_per_t_net_G],
    ["dry_per_t_net (t/t)", dry_per_t_net],
    ["m0_per_t_net_G (t/t)", m0_per_t_net_G],

    ["delta_eps (J/kg)", delta_eps],
    ["kwh_per_kg_ideal", kwh_per_kg_ideal],
    ["kwh_per_kg_input", kwh_per_kg_in],
    ["elec_cost_per_t_lifted (USD/t lifted)", elec_cost_per_t_lifted],
    ["elec_cost_per_t_net_method1 (USD/t net)", elec_cost_per_t_net_method1],
    ["rocket_cost_per_t_net_G (USD/t net)", rocket_cost_per_t_net_G],
    ["method1_cost_per_t_net (USD/t net)", method1_cost_per_t_net],

    ["net_per_port_per_year_elevator (t/yr net)", net_per_port_per_year_elevator],
    ["net_per_port_per_year_launchcap (t/yr net)", net_per_port_per_year_launchcap],
    ["net_per_port_per_year_effective (t/yr net)", net_per_port_per_year_effective],
    ["net_all_ports_per_year_effective (t/yr net)", net_all_ports_per_year_effective],
], columns=["metric", "value"])

summary_df = pd.DataFrame([
    ["Method1 net throughput effective (t/yr)", net_all_ports_per_year_effective],
    ["Method2 net throughput (t/yr)", net_method2_per_year],
], columns=["metric", "value"])

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    inputs_df.to_excel(writer, sheet_name="Inputs", index=False)
    method1_df.to_excel(writer, sheet_name="Method1_Intermediates", index=False)
    df_bases.to_excel(writer, sheet_name="Method2_Bases", index=False)
    summary_df.to_excel(writer, sheet_name="Summary", index=False)

print(f"[OK] Excel exported: {out_path.resolve()}")

# ============================================================
# 4) Multi-objective planning via single weighted objective:
#    Minimize: total_cost + beta * T
#
# beta = per_capita_value * population
# Here: 200,000 USD/person/year * 100,000 people = 2e10 USD/year
# ============================================================

beta = 45e8  # USD/year

# Decision vector x = [m1..mP, n1..nB, T]
n_ports = num_ports
n_bases = len(df_bases)
N = n_ports + n_bases + 1

# Objective: total cost + beta*T
c = np.zeros(N)
c[:n_ports] = method1_cost_per_t_net
c[n_ports:n_ports+n_bases] = df_bases["cost_per_t_net_USD"].to_numpy(dtype=float)
c[-1] = beta

# Equality: total delivered = M
A_eq = np.zeros((1, N))
A_eq[0, :n_ports+n_bases] = 1.0
b_eq = np.array([M], dtype=float)

# Inequalities: capacity constraints
A_ub = []
b_ub = []

# Method 1 port constraints (two caps):
cap_port_elevator = U / kappa_G          # net tons/year/port from elevator
cap_port_launch   = nu_G * L_G           # net tons/year/port from launch cadence & payload cap

for i in range(n_ports):
    # (1) elevator cap: m_i <= cap_port_elevator * T
    row = np.zeros(N)
    row[i] = 1.0
    row[-1] = -cap_port_elevator
    A_ub.append(row)
    b_ub.append(0.0)

    # (2) payload cap via cadence: m_i <= cap_port_launch * T
    row = np.zeros(N)
    row[i] = 1.0
    row[-1] = -cap_port_launch
    A_ub.append(row)
    b_ub.append(0.0)

# Method 2 base capacity: n_j <= payload_per_year_j * T
payload_per_year = df_bases["payload_per_year_t_net"].to_numpy(dtype=float)
for j in range(n_bases):
    row = np.zeros(N)
    row[n_ports + j] = 1.0
    row[-1] = -payload_per_year[j]
    A_ub.append(row)
    b_ub.append(0.0)

A_ub = np.array(A_ub, dtype=float)
b_ub = np.array(b_ub, dtype=float)

# Bounds: all variables >= 0
bounds = [(0, None)] * N

res = linprog(
    c=c,
    A_ub=A_ub, b_ub=b_ub,
    A_eq=A_eq, b_eq=b_eq,
    bounds=bounds,
    method="highs",
)

if not res.success:
    raise RuntimeError("Weighted multi-objective LP failed: " + res.message)

x = res.x
m = x[:n_ports]
n = x[n_ports:n_ports+n_bases]
T = x[-1]

total_cost = method1_cost_per_t_net * m.sum() + np.dot(df_bases["cost_per_t_net_USD"].to_numpy(dtype=float), n)
total_objective = total_cost + beta * T

print("\n===== Weighted multi-objective optimal plan =====")
print(f"beta (USD/year) = {beta:.6e}")
print(f"T (years) = {T:.6f}")
print(f"Total by ports (tons) = {m.sum():.6f}")
print(f"Total by bases (tons) = {n.sum():.6f}")
print(f"Check total (tons) = {m.sum() + n.sum():.6f}")
print(f"Total cost (USD) = {total_cost:.6e}")
print(f"Total objective (USD) = {total_objective:.6e}")

print("\n--- Port allocation m_i (tons) ---")
for i, val in enumerate(m, 1):
    print(f"Port {i}: {val:.6f}")

print("\n--- Base allocation n_j (tons) ---")
for j, val in enumerate(n, 1):
    print(f"{j:02d} {df_bases.loc[j-1,'base']}: {val:.6f}")

# Helpful sanity prints
print("\n--- Sanity checks ---")
print("cap_port_elevator (net/yr/port) =", cap_port_elevator)
print("cap_port_launch   (net/yr/port) =", cap_port_launch)
print("cap_port_effective(net/yr/port) =", min(cap_port_elevator, cap_port_launch))
print("sanity T(3 ports only, effective) =", M / (3 * min(cap_port_elevator, cap_port_launch)))
print("Method2 net throughput (t/yr) =", net_method2_per_year)
print("Total throughput if all used at cap (t/yr) =", 3 * min(cap_port_elevator, cap_port_launch) + net_method2_per_year)
print("sanity T(all caps) =", M / (3 * min(cap_port_elevator, cap_port_launch) + net_method2_per_year))



# ============================================================
# 6) Cumulative cost curves from 2050: Method1 vs Method2 vs Method3
# ============================================================

START_YEAR = 2050

def build_cumulative_curve_const_rates(
    start_year: int,
    total_mass: float,
    yearly_deliveries: np.ndarray,   # tons/year per "channel" (ports/bases)
    cost_per_tons: np.ndarray,       # USD/ton per channel
):
    """
    Assume each channel delivers at constant rate yearly_deliveries[k] (tons/year),
    with per-ton cost cost_per_tons[k].
    Build annual timeline with last year possibly partial.
    Return DataFrame with columns: year, delivered, year_cost, cum_cost, cum_delivered
    """
    rate_total = float(np.sum(yearly_deliveries))
    if rate_total <= 0:
        raise ValueError("Total delivery rate must be > 0")

    # total time in years (continuous)
    T_cont = total_mass / rate_total

    # number of full years
    full_years = int(np.floor(T_cont + 1e-12))
    frac = T_cont - full_years  # remaining fraction of a year (0..1)

    rows = []
    cum_cost = 0.0
    cum_del = 0.0

    # helper to compute one year (or partial year) cost & delivery
    def one_step(year_idx: int, year_fraction: float):
        nonlocal cum_cost, cum_del
        delivered_channels = yearly_deliveries * year_fraction
        delivered = float(np.sum(delivered_channels))
        year_cost = float(np.dot(delivered_channels, cost_per_tons))
        cum_del += delivered
        cum_cost += year_cost
        rows.append({
            "year": start_year + year_idx,
            "delivered_tons": delivered,
            "year_cost_USD": year_cost,
            "cum_delivered_tons": cum_del,
            "cum_cost_USD": cum_cost,
        })

    # full years
    for y in range(full_years):
        one_step(y, 1.0)

    # last partial year (if any)
    if frac > 1e-12:
        one_step(full_years, frac)

    df = pd.DataFrame(rows)
    return df


# ----------------------------
# Method 1 only
# ----------------------------
# Effective per-port cap already computed:
# cap_port_elevator, cap_port_launch, cap_port_effective = min(...)
cap_port_effective = min(cap_port_elevator, cap_port_launch)  # tons/year/port (net)
rate_m1_channels = np.array([cap_port_effective] * num_ports, dtype=float)
cost_m1_channels = np.array([method1_cost_per_t_net] * num_ports, dtype=float)

df_m1_curve = build_cumulative_curve_const_rates(
    start_year=START_YEAR,
    total_mass=M,
    yearly_deliveries=rate_m1_channels,
    cost_per_tons=cost_m1_channels,
)

# ----------------------------
# Method 2 only
# ----------------------------
# Each base runs at its max yearly payload
rate_m2_channels = df_bases["payload_per_year_t_net"].to_numpy(dtype=float)
cost_m2_channels = df_bases["cost_per_t_net_USD"].to_numpy(dtype=float)

df_m2_curve = build_cumulative_curve_const_rates(
    start_year=START_YEAR,
    total_mass=M,
    yearly_deliveries=rate_m2_channels,
    cost_per_tons=cost_m2_channels,
)

# ----------------------------
# Method 3 (LP mix) — use your solved x = [m, n, T]
# ----------------------------
# Assumption: deliver evenly over time horizon T (years)
# Rate per channel = allocated_total / T
if T <= 0:
    raise ValueError("LP solution T must be > 0")

rate_m3_ports = (m / T).astype(float)  # tons/year per port
rate_m3_bases = (n / T).astype(float)  # tons/year per base

rate_m3_channels = np.hstack([rate_m3_ports, rate_m3_bases])
cost_m3_channels = np.hstack([
    np.array([method1_cost_per_t_net] * num_ports, dtype=float),
    df_bases["cost_per_t_net_USD"].to_numpy(dtype=float),
])

df_m3_curve = build_cumulative_curve_const_rates(
    start_year=START_YEAR,
    total_mass=M,
    yearly_deliveries=rate_m3_channels,
    cost_per_tons=cost_m3_channels,
)

# ----------------------------
# Plot cumulative cost vs calendar year
# ----------------------------
plt.figure()
plt.plot(df_m1_curve["year"], df_m1_curve["cum_cost_USD"], marker="o")
plt.plot(df_m2_curve["year"], df_m2_curve["cum_cost_USD"], marker="o")
plt.plot(df_m3_curve["year"], df_m3_curve["cum_cost_USD"], marker="o")

plt.xlabel("Year")
plt.ylabel("Cumulative Cost (USD)")
plt.title("Cumulative Cost from 2050 (Method 1 vs Method 2 vs Method 3)")
plt.grid(True)
plt.legend(["Method 1 only", "Method 2 only", "Method 3 (LP mix)"])
plt.show()

# Optional: also plot cumulative delivered to verify all reach M
plt.figure()
plt.plot(df_m1_curve["year"], df_m1_curve["cum_delivered_tons"], marker="o")
plt.plot(df_m2_curve["year"], df_m2_curve["cum_delivered_tons"], marker="o")
plt.plot(df_m3_curve["year"], df_m3_curve["cum_delivered_tons"], marker="o")
plt.xlabel("Year")
plt.ylabel("Cumulative Delivered (tons)")
plt.title("Cumulative Delivered from 2050 (sanity check)")
plt.grid(True)
plt.legend(["Method 1 only", "Method 2 only", "Method 3 (LP mix)"])
plt.show()

# ----------------------------
# Export curves to Excel
# ----------------------------
with pd.ExcelWriter(out_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    df_m1_curve.to_excel(writer, sheet_name="CumCost_Method1", index=False)
    df_m2_curve.to_excel(writer, sheet_name="CumCost_Method2", index=False)
    df_m3_curve.to_excel(writer, sheet_name="CumCost_Method3", index=False)

print(f"[OK] Cumulative curves written to {out_path.resolve()} (3 sheets)")
print("Method1 end year:", int(df_m1_curve["year"].iloc[-1]), "Total cost:", float(df_m1_curve['cum_cost_USD'].iloc[-1]))
print("Method2 end year:", int(df_m2_curve["year"].iloc[-1]), "Total cost:", float(df_m2_curve['cum_cost_USD'].iloc[-1]))
print("Method3 end year:", int(df_m3_curve["year"].iloc[-1]), "Total cost:", float(df_m3_curve['cum_cost_USD'].iloc[-1]))