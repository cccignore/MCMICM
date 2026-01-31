import math
import pandas as pd
from pathlib import Path

# =========================
# 1) Inputs (from your md + your latest constraints)
# =========================
M = 1e8  # tons
g0 = 9.80665  # m/s^2

# Structural coeff (same for G and E per your instruction)
m_dry = 150.0          # tons
m_payload_ref = 150.0  # tons
alpha = m_dry / m_payload_ref  # alpha_G = alpha_E

# Costs (same for G and E per your instruction)
c_dry = 3.1e6  # USD / ton
c_prop = 5.0e2 # USD / ton

# Method 1: elevator + GEO->Moon stage
dv_G = 2.2e3    # m/s
Isp_G = 380.0   # s
U = 179000.0    # tons/year per port
num_ports = 3

mu = 398600.44 * 1e9  # km^3/s^2 -> m^3/s^2
R_E = 6378e3          # m
r_GEO = 42164e3       # m
eta_E = 0.8
pi_E = 0.03           # USD/kWh

# Method 2: ground direct rocket
dv_base = 15.2e3      # m/s
vrot_equator = 465.1  # m/s
Isp_E = 450.0         # s
L_E = 6000.0          # tons/launch
lambda_j = 2.0        # launches/year/base (your constraint)

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

# =========================
# 2) Method 1 intermediates
# =========================
R_G = math.exp(dv_G / (Isp_G * g0))
kappa_G = R_G * (1 + alpha)

fuel_per_t_net_G = (R_G - 1) * (1 + alpha)  # tons fuel per ton net material
dry_per_t_net = alpha                       # tons dry per ton net material
m0_per_t_net_G = kappa_G                    # tons (at ignition) per ton net

# Elevator energy & cost
delta_eps = mu * (1/R_E - 1/r_GEO)          # J/kg
kwh_per_kg_ideal = delta_eps / 3.6e6
kwh_per_kg_in = kwh_per_kg_ideal / eta_E
elec_cost_per_t_lifted = kwh_per_kg_in * 1000 * pi_E
elec_cost_per_t_net_method1 = elec_cost_per_t_lifted * kappa_G

# GEO->Moon stage rocket cost per ton net
rocket_cost_per_t_net_G = c_dry * dry_per_t_net + c_prop * fuel_per_t_net_G
method1_cost_per_t_net = elec_cost_per_t_net_method1 + rocket_cost_per_t_net_G

# Throughput (net tons/year)
net_per_port_per_year = U / kappa_G
net_all_ports_per_year = num_ports * net_per_port_per_year

# =========================
# 3) Method 2 intermediates (per base)
# =========================
rows = []
for name, lat_deg in bases:
    lat_rad = math.radians(lat_deg)
    vrot = vrot_equator * math.cos(lat_rad)        # m/s
    dv_j = dv_base - vrot                          # m/s
    R_j = math.exp(dv_j / (Isp_E * g0))

    payload_per_launch = L_E / (R_j * (1 + alpha)) # tons net per launch
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

# =========================
# 4) Export to Excel
# =========================
out_path = Path("intermediate_calcs.xlsx")

inputs_df = pd.DataFrame([
    ["M (tons)", M],
    ["g0 (m/s^2)", g0],
    ["alpha (=400/150)", alpha],
    ["c_dry (USD/t)", c_dry],
    ["c_prop (USD/t)", c_prop],
    ["dv_G (m/s)", dv_G],
    ["Isp_G (s)", Isp_G],
    ["U per port (t/yr)", U],
    ["ports", num_ports],
    ["mu (m^3/s^2)", mu],
    ["R_E (m)", R_E],
    ["r_GEO (m)", r_GEO],
    ["eta_E", eta_E],
    ["pi_E (USD/kWh)", pi_E],
    ["dv_base (m/s)", dv_base],
    ["vrot_equator (m/s)", vrot_equator],
    ["Isp_E (s)", Isp_E],
    ["L_E (t/launch)", L_E],
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
    ["elec_cost_per_t_lifted (USD/t)", elec_cost_per_t_lifted],
    ["elec_cost_per_t_net_method1 (USD/t)", elec_cost_per_t_net_method1],
    ["rocket_cost_per_t_net_G (USD/t)", rocket_cost_per_t_net_G],
    ["method1_cost_per_t_net (USD/t)", method1_cost_per_t_net],
    ["net_per_port_per_year (t/yr)", net_per_port_per_year],
    ["net_all_ports_per_year (t/yr)", net_all_ports_per_year],
], columns=["metric", "value"])

summary_df = pd.DataFrame([
    ["Method1 net throughput (t/yr)", net_all_ports_per_year],
    ["Method2 net throughput (t/yr)", net_method2_per_year],
], columns=["metric", "value"])

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    inputs_df.to_excel(writer, sheet_name="Inputs", index=False)
    method1_df.to_excel(writer, sheet_name="Method1_Intermediates", index=False)
    df_bases.to_excel(writer, sheet_name="Method2_Bases", index=False)
    summary_df.to_excel(writer, sheet_name="Summary", index=False)

print(f"Saved: {out_path.resolve()}")