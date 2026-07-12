import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import streamlit as st

from pvt_correlations import (
    oil_gravity_sg, PB_CORRELATIONS, RS_CORRELATIONS, BO_CORRELATIONS,
    vasquez_beggs_co, bo_undersaturated,
    dead_oil_viscosity, saturated_oil_viscosity, undersaturated_oil_viscosity,
    oil_density, oil_density_undersaturated, total_oil_compressibility_saturated,
    VALID_RANGES,
    sutton_pseudocritical, wichert_aziz_correction, z_factor_dak,
    gas_fvf, gas_fvf_bbl, gas_density, gas_viscosity_lee, gas_compressibility,
    GAS_METHOD_NOTES,
)

st.set_page_config(page_title="Black-Oil PVT Calculator", layout="wide")

st.sidebar.title("Reservoir Fluid & Conditions")

API = st.sidebar.number_input("Stock-tank oil gravity, API (deg)", 10.0, 60.0, 35.0, 0.5)
gamma_g_raw = st.sidebar.number_input("Solution gas specific gravity, gamma_g ", 0.55, 1.4, 0.80, 0.01)
T_res = st.sidebar.number_input("Reservoir temperature, T (deg F)", 60.0, 350.0, 180.0, 1.0)
Rsb = st.sidebar.number_input("Solution GOR at bubble point, Rsb (scf/STB)", 10.0, 3000.0, 600.0, 10.0)
P_res = st.sidebar.number_input("Reservoir pressure, P (psia)", 100.0, 12000.0, 5000.0, 50.0)

gamma_o = oil_gravity_sg(API)

st.sidebar.markdown("---")
st.sidebar.subheader("Plot ranges")
P_min, P_max = st.sidebar.slider("Pressure range for P-sensitivity plots (psia)", 14.7, 12000.0, (14.7, max(P_res, 6000.0)))
T_min, T_max = st.sidebar.slider("Temperature range for T-sensitivity plots (deg F)", 60.0, 350.0, (max(T_res - 60, 60.0), T_res + 60))

st.sidebar.markdown("---")
rs_choice = st.sidebar.selectbox(
    "Correlation set driving Rs(P), Bo and viscosity curves",
    list(PB_CORRELATIONS.keys()), index=0,
)
bo_choice = st.sidebar.selectbox(
    "Bo correlation driving oil compressibility & density curves",
    list(BO_CORRELATIONS.keys()), index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Gas properties")
with st.sidebar.expander("Non-hydrocarbon content (sour-gas correction)"):
    y_co2 = st.number_input("CO2 mole fraction", 0.0, 0.6, 0.0, 0.01)
    y_h2s = st.number_input("H2S mole fraction", 0.0, 0.6, 0.0, 0.01)

st.sidebar.markdown("---")
P_fixed_for_T = st.sidebar.number_input(
    "Fixed pressure for all vs-temperature plots (psia)", 14.7, 20000.0, 2500.0, 50.0)

P_array = np.linspace(P_min, P_max, 200)
T_array = np.linspace(T_min, T_max, 200)

gg_for = {"Standing": gamma_g_raw, "Glaso": gamma_g_raw, "Vasquez-Beggs": gamma_g_raw}

pb_results = {name: float(fn(Rsb, gg_for[name], API, T_res)) for name, fn in PB_CORRELATIONS.items()}

pb_vs_T = {name: fn(Rsb, gg_for[name], API, T_array) for name, fn in PB_CORRELATIONS.items()}

# --- Rs sensitivity to pressure (own Pb caps the curve at Rsb) & to temperature ---
rs_vs_P = {name: np.minimum(fn(P_array, gg_for[name], API, T_res), Rsb) for name, fn in RS_CORRELATIONS.items()}
rs_vs_T = {name: np.minimum(fn(P_fixed_for_T, gg_for[name], API, T_array), Rsb) for name, fn in RS_CORRELATIONS.items()}

gg_sel = gg_for[rs_choice]
Pb_sel = pb_results[rs_choice]
Rs_curve_P = np.minimum(RS_CORRELATIONS[rs_choice](P_array, gg_sel, API, T_res), Rsb)

co_sel = vasquez_beggs_co(max(Pb_sel, 1.0), Rsb, T_res, gamma_g_raw, API)
bo_vs_P = {}
for name, fn in BO_CORRELATIONS.items():
    Bo_sat = fn(Rs_curve_P, gg_sel, gamma_o, T_res)
    Bob_at_Pb = fn(Rsb, gg_sel, gamma_o, T_res)
    bo_vs_P[name] = np.where(P_array <= Pb_sel, Bo_sat,
                              bo_undersaturated(Bob_at_Pb, co_sel, P_array, Pb_sel))

bo_vs_T = {name: fn(Rsb, gg_for[rs_choice], gamma_o, T_array) for name, fn in BO_CORRELATIONS.items()}

mu_od_const = dead_oil_viscosity(API, T_res)
mu_ob_curve = saturated_oil_viscosity(Rs_curve_P, mu_od_const)
mu_ob_at_Pb = saturated_oil_viscosity(Rsb, mu_od_const)
mu_vs_P = np.where(P_array <= Pb_sel, mu_ob_curve,
                    undersaturated_oil_viscosity(mu_ob_at_Pb, np.maximum(P_array, 1.0), max(Pb_sel, 1.0)))

mu_od_vs_T = dead_oil_viscosity(API, T_array)
mu_ob_vs_T = saturated_oil_viscosity(Rsb, mu_od_vs_T)

Tpc_sweet, Ppc_sweet = sutton_pseudocritical(gamma_g_raw)
sour_gas = (y_co2 > 0) or (y_h2s > 0)
if sour_gas:
    Tpc, Ppc = wichert_aziz_correction(Tpc_sweet, Ppc_sweet, y_co2, y_h2s)
else:
    Tpc, Ppc = Tpc_sweet, Ppc_sweet

Ppr_vs_P = P_array / Ppc
Tpr_fixedT = (T_res + 460.0) / Tpc
Z_vs_P = z_factor_dak(Ppr_vs_P, np.full_like(P_array, Tpr_fixedT))
Bg_vs_P = gas_fvf(Z_vs_P, T_res, P_array)
Bg_bbl_vs_P = gas_fvf_bbl(Z_vs_P, T_res, P_array)
rho_g_vs_P = gas_density(gamma_g_raw, Z_vs_P, T_res, P_array)
mu_g_vs_P = gas_viscosity_lee(gamma_g_raw, T_res, rho_g_vs_P)
cg_vs_P = gas_compressibility(P_array, Z_vs_P)

Ppr_fixedP = P_fixed_for_T / Ppc
Tpr_vs_T = (T_array + 460.0) / Tpc
Z_vs_T = z_factor_dak(np.full_like(T_array, Ppr_fixedP), Tpr_vs_T)
Bg_vs_T = gas_fvf(Z_vs_T, T_array, P_fixed_for_T)
rho_g_vs_T = gas_density(gamma_g_raw, Z_vs_T, T_array, P_fixed_for_T)
mu_g_vs_T = gas_viscosity_lee(gamma_g_raw, T_array, rho_g_vs_T)

Rs_curve_bo = Rs_curve_P                                        # same Rs(P) trend as Bo/viscosity section
Bo_sat_full = BO_CORRELATIONS[bo_choice](Rs_curve_bo, gg_sel, gamma_o, T_res)   # saturated-oil formula, all P
Bob_choice_at_Pb = BO_CORRELATIONS[bo_choice](Rsb, gg_sel, gamma_o, T_res)
co_choice_at_Pb = vasquez_beggs_co(max(Pb_sel, 1.0), Rsb, T_res, gamma_g_raw, API)

dBo_dP = np.gradient(Bo_sat_full, P_array)
dRs_dP = np.gradient(Rs_curve_bo, P_array)
co_saturated = total_oil_compressibility_saturated(Bo_sat_full, dBo_dP, Bg_bbl_vs_P, dRs_dP)
co_undersaturated = vasquez_beggs_co(np.maximum(P_array, 1.0), Rsb, T_res, gamma_g_raw, API)
co_vs_P = np.where(P_array <= Pb_sel, co_saturated, co_undersaturated)

rho_o_sat_full = oil_density(Rs_curve_bo, gg_sel, gamma_o, Bo_sat_full)
rho_ob_choice = oil_density(Rsb, gg_sel, gamma_o, Bob_choice_at_Pb)
rho_o_vs_P = np.where(P_array <= Pb_sel, rho_o_sat_full,
                       oil_density_undersaturated(rho_ob_choice, co_choice_at_Pb, P_array, Pb_sel))

co_vs_T = vasquez_beggs_co(max(P_fixed_for_T, 1.0), Rsb, T_array, gamma_g_raw, API)
Rs_for_rho_T = rs_vs_T[rs_choice]
Bo_for_rho_T = BO_CORRELATIONS[bo_choice](Rs_for_rho_T, gg_sel, gamma_o, T_array)
rho_o_vs_T = oil_density(Rs_for_rho_T, gg_sel, gamma_o, Bo_for_rho_T)

def check_ranges(name, API, gamma_g, T, Rs):
    rng = VALID_RANGES[name]
    msgs = []
    checks = [("API gravity", API, rng["API"], "deg API"),
              ("gas gravity", gamma_g, rng["gamma_g"], ""),
              ("temperature", T, rng["T"], "deg F"),
              ("Rs", Rs, rng["Rs"], "scf/STB")]
    for label, val, (lo, hi), unit in checks:
        if not (lo <= val <= hi):
            msgs.append(
                f"**{name}**: {label} = {val:.2f}{unit} is outside the correlation's "
                f"original data range ({lo}-{hi}{unit}).; "
                f"extrapolating beyond that calibration range can bias the result."
            )
    return msgs


anomalies = []

for name in PB_CORRELATIONS:
    anomalies += [("warning", m) for m in check_ranges(name, API, gg_for[name], T_res, Rsb)]

pbs = np.array(list(pb_results.values()))
pb_spread = (pbs.max() - pbs.min()) / pbs.mean() * 100
if pb_spread > 25:
    anomalies.append(("warning",
        f"The three bubble-point correlations disagree by {pb_spread:.0f}% "
        f"(Standing = {pb_results['Standing']:.0f}, Vasquez-Beggs = {pb_results['Vasquez-Beggs']:.0f}, "
        f"Glaso = {pb_results['Glaso']:.0f} psia). Each correlation was regressed from crude oils "
        "from a different producing region (California, Gulf Coast, and North Sea respectively). "
        "A large spread usually means the input oil/gas properties sit far from at least one of "
        "those calibration datasets, so that correlation is being extrapolated."))

for name, curve in rs_vs_P.items():
    if np.any(np.diff(curve) < -1e-6):
        anomalies.append(("error",
            f"{name}'s Rs(P) curve is not monotonically increasing with pressure. Physically, "
            "solution gas-oil ratio can only increase (or stay flat once the oil is fully "
            "saturated) as pressure rises below the bubble point, since more gas dissolves "
            "into the oil under greater pressure. A decrease indicates the correlation is "
            "being evaluated far outside its valid range."))

if co_sel < 0:
    anomalies.append(("warning",
        f"The computed isothermal oil compressibility above the bubble point is negative "
        f"(co = {co_sel:.2e} 1/psi). This is not physical - oil should always shrink under "
        "added pressure. It usually results from an unusually low/high API or gas gravity "
        "combination pushing the Vasquez-Beggs compressibility correlation outside its "
        "intended range."))

if np.any(np.diff(mu_od_vs_T) > 1e-9):
    anomalies.append(("warning",
        "Dead-oil viscosity is predicted to increase with temperature somewhere in the chosen "
        "range. Physically, oil viscosity should decrease monotonically as temperature rises "
        "because thermal agitation reduces intermolecular friction; a rise usually signals the "
        "Beggs-Robinson dead-oil correlation is being extrapolated to an unusually low or high "
        "API gravity."))

if co_choice_at_Pb < 0:
    anomalies.append(("warning",
        f"The computed isothermal oil compressibility above the bubble point is negative "
        f"(co = {co_choice_at_Pb:.2e} 1/psi) for the '{bo_choice}' Bo curve. This is not "
        "physical - oil should always shrink under added pressure. It usually results from an "
        "unusually low/high API or gas gravity combination pushing the Vasquez-Beggs "
        "compressibility correlation outside its intended range."))

Tpr_check = (T_res + 460.0) / Tpc
Ppr_check = P_res / Ppc
if not (1.0 <= Tpr_check <= 3.0) or not (0.2 <= Ppr_check <= 30):
    anomalies.append(("warning",
        f"At the reservoir condition, the pseudo-reduced temperature/pressure (Tpr = "
        f"{Tpr_check:.2f}, Ppr = {Ppr_check:.2f}) falls outside the range the "
        "Dranchuk-Abou-Kassem Z-factor correlation was fitted to (Tpr 1.0-3.0, Ppr 0.2-30). "
        "The Z-factor, and everything derived from it (Bg, gas density, gas viscosity, gas "
        "compressibility), may be unreliable at this condition."))

if np.any(Z_vs_P < 0.1) or np.any(Z_vs_P > 2.0):
    anomalies.append(("error",
        "The computed gas Z-factor leaves the physically expected range (roughly 0.1-2.0) "
        "somewhere in the pressure sweep. Real gases compress smoothly with pressure at a "
        "given temperature; a Z-factor this far from unity usually means the pseudo-critical "
        "properties (derived from gas gravity) do not represent this gas well, or that the "
        "iterative solver has not converged."))

if np.any(np.diff(Bg_vs_P) > 1e-9):
    anomalies.append(("warning",
        "Gas FVF (Bg) is predicted to increase with pressure somewhere in the sweep. "
        "Physically, gas is highly compressible, so Bg should decrease monotonically as "
        "pressure rises at constant temperature. An increase points to a Z-factor that is "
        "not behaving smoothly with pressure, most likely because Ppr/Tpr have left the "
        "correlation's valid range."))

if np.any(np.diff(rho_g_vs_P) < -1e-9):
    anomalies.append(("warning",
        "Gas density is predicted to decrease with pressure somewhere in the sweep. Since gas "
        "mass is conserved while its volume shrinks under pressure, density should rise "
        "monotonically with pressure at constant temperature; a decrease again traces back to "
        "an unreliable Z-factor at that condition."))

if np.any(cg_vs_P < 0):
    anomalies.append(("warning",
        "The computed gas compressibility (cg) is negative somewhere in the pressure sweep. "
        "Gas compressibility should always be positive; a negative value here comes from "
        "numerical noise in dZ/dP where the Z-factor curve is not smooth, usually because "
        "part of the sweep lies outside the Z-factor correlation's valid Ppr/Tpr range."))

if not anomalies:
    anomalies.append(("success", "No anomalies detected - all curves behave as expected from black-oil theory over the selected ranges."))


st.title("Black-Oil PVT Calculator")
st.caption(
    "Estimates bubble-point pressure, solution gas-oil ratio, oil formation volume factor and "
    "oil viscosity from empirical black-oil correlations, and checks the results for "
    "physically inconsistent behaviour."
)

(tab_overview, tab_pb, tab_rs, tab_bo, tab_visc, tab_ocomp,
 tab_gas, tab_anom, tab_ref) = st.tabs(
    ["Overview", "Bubble-Point Pressure", "Gas Solubility (Rs)",
     "Oil FVF (Bo)", "Oil Viscosity", "Oil Compressibility & Density",
     "Gas Properties", "Anomaly Check", "Correlation Reference"]
)

def dual_plot(x1, y1_dict, xlabel1, x2, y2_dict, xlabel2, ylabel, title, vline1=None, hline1=None):
    """Two-panel figure: property vs. x1 (left) and vs. x2 (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for name, curve in y1_dict.items():
        axes[0].plot(x1, curve, label=name)
    if vline1 is not None:
        axes[0].axvline(vline1, ls="--", color="gray", lw=1)
    if hline1 is not None:
        axes[0].axhline(hline1, ls="--", color="gray", lw=1)
    axes[0].set_xlabel(xlabel1)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(f"{title} vs. {xlabel1}")
    if len(y1_dict) > 1:
        axes[0].legend(fontsize=8)
    for name, curve in y2_dict.items():
        axes[1].plot(x2, curve, label=name)
    axes[1].set_xlabel(xlabel2)
    axes[1].set_title(f"{title} vs. {xlabel2}")
    if len(y2_dict) > 1:
        axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig

with tab_overview:
    c1, c2, c3 = st.columns(3)
    c1.metric("Stock-tank oil gravity, gamma_o", f"{gamma_o:.4f}")
    c2.metric("Gas gravity used (VB, corrected)" , f"{gamma_g_raw:.4f}")
    c3.metric("Active Rs/Pb correlation set", rs_choice)

    st.subheader("Bubble-point pressure")
    df_pb = pd.DataFrame({"Correlation": list(pb_results.keys()),
                           "Bubble-Point Pressure (psia)": [round(v, 1) for v in pb_results.values()]})
    st.dataframe(df_pb, hide_index=True, width='stretch')

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(df_pb["Correlation"], df_pb["Bubble-Point Pressure (psia)"], color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylabel("Pb (psia)")
    ax.set_title("Bubble-point pressure by correlation")
    st.pyplot(fig, clear_figure=True)

    st.info(
        f"At the base case (T = {T_res:.0f} deg F, Rsb = {Rsb:.0f} scf/STB), the selected "
        f"set is **{rs_choice}**, giving Pb = {Pb_sel:.0f} psia. "
        f"{'The reservoir pressure is above Pb (undersaturated oil).' if P_res > Pb_sel else 'The reservoir pressure is at or below Pb (saturated oil, free gas present).'}"
    )

with tab_pb:
    st.subheader("Bubble-Point Pressure, Pb")
    st.markdown("Computed from the input solution GOR at bubble point, **Rsb**.")
    st.dataframe(df_pb, hide_index=True, width='stretch')

    st.markdown("#### Sensitivity to temperature (Rsb fixed)")
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, curve in pb_vs_T.items():
        ax.plot(T_array, curve, label=name)
    ax.set_xlabel("Temperature (deg F)")
    ax.set_ylabel("Pb (psia)")
    ax.set_title("Bubble-point pressure vs. temperature")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

with tab_rs:
    st.subheader("Solution Gas-Oil Ratio, Rs")
    st.caption("Curves are capped at Rsb, since no additional gas can dissolve once the oil is saturated.")

    st.markdown("#### Sensitivity to pressure (T fixed at reservoir temperature)")
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, curve in rs_vs_P.items():
        ax.plot(P_array, curve, label=name)
    ax.axhline(Rsb, ls="--", color="gray", lw=1, label="Rsb (input)")
    ax.set_xlabel("Pressure (psia)")
    ax.set_ylabel("Rs (scf/STB)")
    ax.set_title("Gas solubility vs. pressure")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    st.markdown(f"#### Sensitivity to temperature (P fixed at {P_fixed_for_T:.0f} psia)")
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, curve in rs_vs_T.items():
        ax.plot(T_array, curve, label=name)
    ax.set_xlabel("Temperature (deg F)")
    ax.set_ylabel("Rs (scf/STB)")
    ax.set_title("Gas solubility vs. temperature")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

with tab_bo:
    st.subheader("Oil Formation Volume Factor, Bo")
    st.caption(f"Rs(P) driven by the **{rs_choice}** correlation set. Dashed line marks Pb = {Pb_sel:.0f} psia; "
               "above it, Bo declines due to compression of undersaturated oil (Vasquez-Beggs compressibility).")

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, curve in bo_vs_P.items():
        ax.plot(P_array, curve, label=name)
    ax.axvline(Pb_sel, ls="--", color="gray", lw=1)
    ax.set_xlabel("Pressure (psia)")
    ax.set_ylabel("Bo (bbl/STB)")
    ax.set_title("Oil FVF vs. pressure")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    st.markdown("#### Sensitivity to temperature (evaluated at Rs = Rsb)")
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, curve in bo_vs_T.items():
        ax.plot(T_array, curve, label=name)
    ax.set_xlabel("Temperature (deg F)")
    ax.set_ylabel("Bo (bbl/STB)")
    ax.set_title("Oil FVF vs. temperature")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

with tab_visc:
    st.subheader("Oil Viscosity")
    st.caption("Beggs-Robinson for dead-oil and saturated viscosity, extended above the bubble "
               "point with the Vasquez-Beggs' undersaturated-oil relation. ")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(P_array, mu_vs_P, color="#C44E52")
    ax.axvline(Pb_sel, ls="--", color="gray", lw=1, label="Pb")
    ax.set_xlabel("Pressure (psia)")
    ax.set_ylabel("Oil viscosity (cp)")
    ax.set_title(f"Oil viscosity vs. pressure (T = {T_res:.0f} deg F)")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    st.markdown("#### Sensitivity to temperature")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(T_array, mu_od_vs_T, label="Dead oil (gas-free)")
    ax.plot(T_array, mu_ob_vs_T, label="Saturated (at Rsb)")
    ax.set_xlabel("Temperature (deg F)")
    ax.set_ylabel("Oil viscosity (cp)")
    ax.set_title("Oil viscosity vs. temperature")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

with tab_ocomp:
    st.subheader("Oil Compressibility, co")
    st.caption(
        f"Below Pb: total two-phase compressibility from the '{bo_choice}' Bo(P) and "
        f"'{rs_choice}' Rs(P) trends, co = -(1/Bo)(dBo/dP) + (Bg/Bo)(dRs/dP). "
        "Above Pb: Vasquez-Beggs (1980) single-phase relation."
    )
    fig = dual_plot(P_array, {"co": co_vs_P}, "Pressure (psia)",
                     T_array, {"co (undersaturated, VB)": co_vs_T}, "Temperature (deg F)",
                     "Compressibility (1/psi)", "Oil compressibility", vline1=Pb_sel)
    st.pyplot(fig, clear_figure=True)

    st.subheader("Oil Density, rho_o")
    st.caption(f"From material balance: rho_o = (62.4 gamma_o + 0.0136 Rs gamma_g) / Bo, "
               f"using the '{bo_choice}' Bo correlation.")
    fig = dual_plot(P_array, {"rho_o": rho_o_vs_P}, "Pressure (psia)",
                     T_array, {"rho_o": rho_o_vs_T}, "Temperature (deg F)",
                     "Oil density (lbm/ft3)", "Oil density", vline1=Pb_sel)
    st.pyplot(fig, clear_figure=True)

with tab_gas:
    st.subheader("Gas Properties")
    note = (f"Pseudo-criticals: Tpc = {Tpc:.1f} R, Ppc = {Ppc:.1f} psia "
            f"({'Wichert-Aziz sour-gas corrected, ' if sour_gas else ''}Sutton). "
            "Z-factor: Dranchuk-Abou-Kassem. Viscosity: Lee-Gonzalez-Eakin.")
    st.caption(note)

    st.markdown("#### Z-factor (gas deviation factor)")
    fig = dual_plot(P_array, {"Z": Z_vs_P}, "Pressure (psia)",
                     T_array, {"Z": Z_vs_T}, "Temperature (deg F)",
                     "Z (dimensionless)", "Gas Z-factor")
    st.pyplot(fig, clear_figure=True)

    st.markdown("#### Gas formation volume factor, Bg")
    fig = dual_plot(P_array, {"Bg": Bg_vs_P}, "Pressure (psia)",
                     T_array, {"Bg": Bg_vs_T}, "Temperature (deg F)",
                     "Bg (rcf/scf)", "Gas FVF")
    st.pyplot(fig, clear_figure=True)

    st.markdown("#### Gas viscosity")
    fig = dual_plot(P_array, {"mu_g": mu_g_vs_P}, "Pressure (psia)",
                     T_array, {"mu_g": mu_g_vs_T}, "Temperature (deg F)",
                     "Gas viscosity (cp)", "Gas viscosity")
    st.pyplot(fig, clear_figure=True)

    st.markdown("#### Gas density")
    fig = dual_plot(P_array, {"rho_g": rho_g_vs_P}, "Pressure (psia)",
                     T_array, {"rho_g": rho_g_vs_T}, "Temperature (deg F)",
                     "Gas density (lbm/ft3)", "Gas density")
    st.pyplot(fig, clear_figure=True)

    st.markdown("#### Gas compressibility, cg")
    st.caption("From a numerical derivative of the Z(P) curve at fixed T; shown vs. pressure only.")
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(P_array, cg_vs_P, color="#8172B2")
    ax.set_xlabel("Pressure (psia)")
    ax.set_ylabel("cg (1/psi)")
    ax.set_title("Gas compressibility vs. pressure")
    st.pyplot(fig, clear_figure=True)

with tab_anom:
    st.subheader("Anomaly Check")
    st.caption("Automated checks of the computed values and curves against black-oil PVT theory.")
    for level, msg in anomalies:
        if level == "error":
            st.error(msg)
        elif level == "warning":
            st.warning(msg)
        elif level == "success":
            st.success(msg)
        else:
            st.info(msg)

with tab_ref:
    st.subheader("Correlation Applicability Ranges")
    st.caption("Approximate ranges of the original regression datasets. Values outside these ranges "
               "are extrapolations and should be treated with caution.")
    rows = []
    for name, rng in VALID_RANGES.items():
        rows.append({
            "Correlation": name,
            "API (deg)": f"{rng['API'][0]}-{rng['API'][1]}",
            "Gas gravity": f"{rng['gamma_g'][0]}-{rng['gamma_g'][1]}",
            "Temperature (F)": f"{rng['T'][0]}-{rng['T'][1]}",
            "Rs (scf/STB)": f"{rng['Rs'][0]}-{rng['Rs'][1]}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
