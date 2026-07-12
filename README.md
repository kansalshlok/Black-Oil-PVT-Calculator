# Black-Oil PVT Calculator

A Streamlit dashboard for black-oil PVT property estimation.

## What it does
Fifteen black-oil PVT properties, each with individual pressure- and
temperature-sensitivity plots:

**Oil**
- Bubble-point pressure (Pb): Standing, Vasquez-Beggs, Glaso
- Solution gas-oil ratio (Rs): Standing, Vasquez-Beggs, Glaso
- Formation volume factor (Bo): Standing, Glaso, Al-Marhoun, Petrosky-Farshad,
  extended above Pb with Vasquez-Beggs undersaturated compressibility
- Viscosity: Beggs-Robinson (dead-oil & saturated) + Vasquez-Beggs (undersaturated)
- Isothermal compressibility (co): the rigorous two-phase definition
  co = -(1/Bo)(dBo/dP) + (Bg/Bo)(dRs/dP) below Pb, Vasquez-Beggs above Pb
- Density: black-oil material balance, rho_o = (62.4 gamma_o + 0.0136 Rs gamma_g)/Bo

**Gas**
- Z-factor: Dranchuk & Abou-Kassem (1975), with Sutton (1985) pseudo-criticals
  and an optional Wichert & Aziz (1972) sour-gas (CO2/H2S) correction
- Formation volume factor (Bg)
- Viscosity: Lee-Gonzalez-Eakin (1966)
- Isothermal compressibility (cg): numerical derivative of the Z(P) curve
- Density

An automated "Anomaly Check" tab flags out-of-range inputs, non-physical
trends (Rs decreasing with pressure, Bo/density rising instead of falling
above Pb, a missing viscosity minimum at Pb, Z-factor or Bg behaving
unphysically, negative compressibilities, water properties drifting outside
normal ranges), and explains the likely theoretical cause for each.

## Files
- `app.py` — the Streamlit dashboard (all UI, plotting, and anomaly logic)
- `pvt_correlations.py` — the underlying correlation formulas, kept separate
  and dependency-light (only numpy) so they can be reused or tested on their own
- `requirements.txt` — the libraries needed (streamlit, numpy, pandas for
  tables, matplotlib for plots — no other dependencies)

## Notes on the correlations
- Every oil correlation was regressed from a specific set of crude oils (see
  the "Correlation Reference" tab in the app for applicability ranges and
  dataset origin). Predictions outside those ranges are extrapolations — the
  dashboard warns you when this happens.
- Above the bubble point, Bo, density and viscosity are extended using the
  Vasquez-Beggs isothermal-compressibility and undersaturated-viscosity
  relations, since the other oil correlations only strictly apply to
  saturated oil.
- The "Correlation set driving Rs(P), Bo and viscosity curves" selector in the
  sidebar lets you pick which Rs(P) trend feeds the Bo, compressibility,
  density and viscosity calculations, so results are compared on a
  consistent Rs basis.
- The gas properties assume a sweet (CO2/H2S-free) gas unless you enter
  non-zero CO2/H2S mole fractions in the sidebar, in which case the
  Wichert-Aziz correction is applied automatically.
- Water properties assume the brine is essentially gas-free, which is the
  standard assumption for a black-oil water leg away from the gas cap.
