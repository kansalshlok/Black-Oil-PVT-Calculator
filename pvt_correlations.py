import numpy as np

def oil_gravity_sg(API):
    return 141.5 / (131.5 + API)

def standing_pb(Rs, gamma_g, API, T):
    return 18.2 * ((Rs / gamma_g) ** 0.83 * 10 ** (0.00091 * T - 0.0125 * API) - 1.4)

def vb_constants(API):
    if API <=30:
        return (0.0362, 1.0937, 25.724) 
    else: 
        return (0.0178, 1.1870, 23.931)

def vasquez_beggs_pb(Rs, gamma_gs, API, T):
    C1, C2, C3 = vb_constants(API)
    return (Rs / (C1 * gamma_gs * np.exp(C3 * API / (T + 460.0)))) ** (1.0 / C2)


def glaso_pb(Rs, gamma_g, API, T):
    pbstar = (Rs / gamma_g) ** 0.816 * (T ** 0.172 / API ** 0.989)
    log_pb = 1.7669 + 1.7447 * np.log10(pbstar) - 0.30218 * (np.log10(pbstar)) ** 2
    return 10 ** log_pb

def standing_rs(P, gamma_g, API, T):
    return gamma_g * ((P / 18.2 + 1.4) * 10 ** (-0.00091 * T + 0.0125 * API)) ** 1.204

def vasquez_beggs_rs(P, gamma_gs, API, T):
    C1, C2, C3 = vb_constants(API)
    return C1 * gamma_gs * P ** C2 * np.exp(C3 * API / (T + 460.0))

def glaso_rs(P, gamma_g, API, T):
    x = 2.8869 - (14.1811 - 3.3093 * np.log10(P))**0.5
    pbstar = 10 ** x
    return gamma_g * (pbstar * API ** 0.989 / T ** 0.172) ** 1.2255

PB_CORRELATIONS = {
    "Standing":        standing_pb,
    "Vasquez-Beggs":   vasquez_beggs_pb,
    "Glaso":           glaso_pb,
}
RS_CORRELATIONS = {
    "Standing":        standing_rs,
    "Vasquez-Beggs":   vasquez_beggs_rs,
    "Glaso":           glaso_rs,
}

def standing_bo(Rs, gamma_g, gamma_o, T):
    F = Rs * (gamma_g / gamma_o) ** 0.5 + 1.25 * T
    return 0.9759 + 0.00012 * F ** 1.2

def marhoun_bo(Rs, gamma_g, gamma_o, T):
    T_R = T + 460.0
    F = Rs ** 0.742390 * gamma_g ** 0.323294 * gamma_o ** (-1.202040)
    return 0.497069 + 0.862963e-3 * T_R + 0.182594e-2 * F + 0.318099e-5 * F ** 2

def petrosky_bo(Rs, gamma_g, gamma_o, T):
    F = Rs ** 0.3738 * (gamma_g ** 0.2914 / gamma_o ** 0.6265) + 0.24626 * T ** 0.5371
    return 1.0113 + 7.2046e-5 * F ** 3.0936


BO_CORRELATIONS = {
    "Standing":          standing_bo,
    "Marhoun":           marhoun_bo,
    "Petrosky-Farshad":  petrosky_bo,
}

def vasquez_beggs_co(P, Rsb, T, gamma_gs, API):
    return (-1433 + 5 * Rsb + 17.2 * T - 1180 * gamma_gs + 12.61 * API) / (1e5 * P)


def bo_undersaturated(Bob, co, P, Pb):
    return Bob * np.exp(-co * (P - Pb))

def dead_oil_viscosity(API, T):
    X = 10 ** (3.0324 - 0.02023 * API) * T ** (-1.163)
    return 10 ** X - 1


def saturated_oil_viscosity(Rs, mu_od):
    A = 10.715 * (Rs + 100) ** (-0.515)
    B = 5.44 * (Rs + 150) ** (-0.338)
    return A * mu_od ** B

def undersaturated_oil_viscosity(mu_ob, P, Pb):
    m = 2.6 * P ** 1.187 * np.exp(-11.513 - 8.98e-5 * P)
    return mu_ob * (P / Pb) ** m

VALID_RANGES = {
    "Standing":          {"API": (16.5, 63.8), "gamma_g": (0.59, 0.95), "T": (100, 258), "Rs": (20, 1425)},
    "Vasquez-Beggs":     {"API": (15.3, 59.5), "gamma_g": (0.51, 1.35), "T": (75, 294),  "Rs": (20, 2070)},
    "Glaso":             {"API": (22.3, 48.1), "gamma_g": (0.65, 1.28), "T": (80, 280),  "Rs": (90, 2637)},
    "Marhoun":           {"API": (19.4, 44.6), "gamma_g": (0.75, 1.02), "T": (74, 240),  "Rs": (26, 1602)},
    "Petrosky-Farshad":  {"API": (16.3, 45.0), "gamma_g": (0.58, 0.87), "T": (114, 288), "Rs": (217, 1406)},
}

def total_oil_compressibility_saturated(Bo, dBo_dP, Bg_res_bbl_per_scf, dRs_dP):
    return -(1.0 / Bo) * dBo_dP + (Bg_res_bbl_per_scf / Bo) * dRs_dP

def oil_density(Rs, gamma_g, gamma_o, Bo):
    return (62.4 * gamma_o + 0.0136 * Rs * gamma_g) / Bo


def oil_density_undersaturated(rho_ob, co, P, Pb):
    return rho_ob * np.exp(co * (P - Pb))

def sutton_pseudocritical(gamma_g):
    Tpc = 169.2 + 349.5 * gamma_g - 74.0 * gamma_g ** 2
    Ppc = 756.8 - 131.0 * gamma_g - 3.6 * gamma_g ** 2
    return Tpc, Ppc


def wichert_aziz_correction(Tpc, Ppc, y_co2, y_h2s):
    A = y_co2 + y_h2s
    B = y_h2s
    eps = 120 * (A ** 0.9 - A ** 1.6) + 15 * (B ** 0.5 - B ** 4)
    Tpc_corr = Tpc - eps
    Ppc_corr = Ppc * Tpc_corr / (Tpc + B * (1 - B) * eps)
    return Tpc_corr, Ppc_corr


DAK_constants = dict(A1=0.3265, A2=-1.0700, A3=-0.5339, A4=0.01569, A5=-0.05165,
            A6=0.5475, A7=-0.7361, A8=0.1844, A9=0.1056, A10=0.6134, A11=0.7210)


def z_factor_dak(Ppr, Tpr, max_iter=100, tol=1e-10):
    Ppr = np.asarray(Ppr, dtype=float)
    Tpr = np.asarray(Tpr, dtype=float)
    Ppr, Tpr = np.broadcast_arrays(Ppr, Tpr)
    Ppr = Ppr.astype(float).copy()
    Tpr = np.maximum(Tpr.astype(float).copy(), 1.01)
 
    a = DAK_constants
    Z = np.ones_like(Ppr)
    for _ in range(max_iter):
        rho_r = 0.27 * Ppr / (Z * Tpr)
        Z_new = (1
                 + (a["A1"] + a["A2"] / Tpr + a["A3"] / Tpr ** 3 + a["A4"] / Tpr ** 4 + a["A5"] / Tpr ** 5) * rho_r
                 + (a["A6"] + a["A7"] / Tpr + a["A8"] / Tpr ** 2) * rho_r ** 2
                 - a["A9"] * (a["A7"] / Tpr + a["A8"] / Tpr ** 2) * rho_r ** 5
                 + a["A10"] * (1 + a["A11"] * rho_r ** 2) * (rho_r ** 2 / Tpr ** 3) * np.exp(-a["A11"] * rho_r ** 2))
        if np.max(np.abs(Z_new - Z)) < tol:
            Z = Z_new
            break
        Z = 0.5 * (Z + Z_new) 
    return Z

def gas_fvf(Z, T, P):
    return 0.02827 * Z * (T + 460.0) / P


def gas_fvf_bbl(Z, T, P):
    return 0.00504 * Z * (T + 460.0) / P


def gas_density(gamma_g, Z, T, P):
    return 2.70 * gamma_g * P / (Z * (T + 460.0))


def gas_viscosity_lee(gamma_g, T, rho_g_lbft3):
    T_R = T + 460.0
    M = 28.97 * gamma_g
    K = (9.4 + 0.02 * M) * T_R ** 1.5 / (209 + 19 * M + T_R)
    X = 3.5 + 986.0 / T_R + 0.01 * M
    Y = 2.4 - 0.2 * X
    rho_g_cc = rho_g_lbft3 / 62.4
    return 1e-4 * K * np.exp(X * rho_g_cc ** Y)


def gas_compressibility(P_array, Z_array):
    P_array = np.asarray(P_array, dtype=float)
    Z_array = np.asarray(Z_array, dtype=float)
    dZ_dP = np.gradient(Z_array, P_array)
    return 1.0 / P_array - (1.0 / Z_array) * dZ_dP

GAS_METHOD_NOTES = {
    "Pseudo-critical properties": "Sutton, from gas specific gravity",
    "Sour-gas correction":        "Wichert & Aziz, for CO2 / H2S content (optional)",
    "Z-factor":                   "Dranchuk & Abou-Kassem, a close fit to the Standing-Katz "
                                   "chart, valid for 0.2 <= Ppr <= 30 and 1.0 <= Tpr <= 3.0",
    "Gas viscosity":              "Lee, Gonzalez & Eakin",
}

