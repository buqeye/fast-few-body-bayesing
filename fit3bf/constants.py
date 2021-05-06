import numpy as np

param_names_chiral = {
    "LO": ["Ct_1S0np", "Ct_1S0nn", "Ct_1S0pp", "Ct_3S1"],
    "NLO": [
        "Ct_1S0np",
        "Ct_1S0nn",
        "Ct_1S0pp",
        "Ct_3S1",
        "C_1S0",
        "C_3P0",
        "C_1P1",
        "C_3P1",
        "C_3S1",
        "C_3S1-3D1",
        "C_3P2",
    ],
    "NNLO": [
        "c1",
        "c3",
        "c4",
        "Ct_1S0np",
        "Ct_1S0nn",
        "Ct_1S0pp",
        "Ct_3S1",
        "C_1S0",
        "C_3P0",
        "C_1P1",
        "C_3P1",
        "C_3S1",
        "C_3S1-3D1",
        "C_3P2",
        "c_D",
        "c_E",
    ],
}

param_names_vary = {
    "3N": ["c_D", "c_E"],
    "NN": [
        "Ct_1S0np",
        "Ct_1S0nn",
        "Ct_1S0pp",
        "Ct_3S1",
        "C_1S0",
        "C_3P0",
        "C_1P1",
        "C_3P1",
        "C_3S1",
        "C_3S1-3D1",
        "C_3P2",
    ],
    "piN": ["c1", "c3", "c4"],
}

param_names_tex_map = {
    "Ct_1S0np": r"$\tilde C^{np}_{1S0}$",
    "Ct_1S0nn": r"$\tilde C^{nn}_{1S0}$",
    "Ct_1S0pp": r"$\tilde C^{pp}_{1S0}$",
    "Ct_3S1": r"$\tilde C_{3S1}$",
    "C_1S0": "$C_{1S0}$",
    "C_3P0": "$C_{3P0}$",
    "C_1P1": "$C_{1P1}$",
    "C_3P1": "$C_{3P1}$",
    "C_3S1": "$C_{3S1}$",
    #     "C_3S1-3D1": r"$C_{3S1\mbox{--}3D1}$",
    "C_3S1-3D1": r"$C_{3S1}^{3D1}$",
    "C_3P2": "$C_{3P2}$",
    "c_D": "$c_D$",
    "c_E": "$c_E$",
    "cbar": r"$\bar c$",
    "Q": r"$Q$",
}

obs_names_tex_map = {
    'H3': r'$E({}^3{\rm H})$',
    'He3': r'$E(^{}^3{\rm He})$',
    'He4': r'$E({}^4{\rm He})$',
    'He4-radius': r'$r({}^4{\rm He})$',
    #     'H3-halflife': r'$\langle E_{1}^{\rm A} \rangle$',
    'H3-halflife': r'$fT_{1/2}$',
}

nnlo_sat_lecs = {
    "c1": -1.12152119963259,
    "c3": -3.92500585648682,
    "c4": +3.76568715858592,
    "Ct_1S0np": -0.15982244957832,
    "Ct_1S0nn": -0.15915026828018,
    "Ct_1S0pp": -0.15814937937011,
    "Ct_3S1": -0.17767436449900,
    "C_1S0": +2.53936778505038,
    "C_3P0": +1.39836559187614,
    "C_1P1": +0.55595876513335,
    "C_3P1": -1.13609526332782,
    "C_3S1": +1.00289267348351,
    "C_3S1-3D1": +0.60071604833596,
    "C_3P2": -0.80230029533846,
    "c_D": +0.81680589148271,
    "c_E": -0.03957471270351,
}

nnlo_450_lecs = {
    "c1": -0.7424161689888585,
    "c3": -3.6126853832948798,
    "c4": 2.4390225292517886,
    "Ct_1S0np": -0.15263259,
    "Ct_1S0nn": -0.15231072,
    "Ct_1S0pp": -0.1519951,
    "Ct_3S1": -0.17844706,
    "C_1S0": 2.39169697,
    "C_3P0": 0.99935281,
    "C_1P1": 0.22117731,
    "C_3P1": -0.97352638,
    "C_3S1": 0.55144146,
    "C_3S1-3D1": 0.43702288,
    "C_3P2": -0.69226719,
    "c_D": 0.1,
    "c_E": 0.1,
}

# in same order as keys in dict nnlo_450_lecs
nnlo_450_cov = np.asarray([
    [6.22e-09, 5.91e-09, 5.49e-09, -6.07e-09, -1.39e-07, 8.23e-08, -7.08e-08, 9.60e-08, 6.01e-08, 3.05e-08, -6.13e-09],
    [5.91e-09, 9.03e-08, 5.30e-09, -6.43e-09, -1.34e-07, 7.99e-08, -6.79e-08, 9.30e-08, 6.10e-08, 2.92e-08, -5.88e-09],
    [5.49e-09, 5.30e-09, 4.95e-09, -5.96e-09, -1.24e-07, 7.41e-08, -6.30e-08, 8.62e-08, 5.65e-08, 2.70e-08, -5.42e-09],
    [-6.07e-09, -6.43e-09, -5.96e-09, 5.78e-07, 1.51e-07, -5.99e-07, -2.11e-06, -3.68e-07, -3.90e-06, -4.40e-06, 3.93e-08],
    [-1.39e-07, -1.34e-07, -1.24e-07, 1.51e-07, 3.14e-06, -1.87e-06, 1.59e-06, -2.18e-06, -1.43e-06, -6.85e-07, 1.38e-07],
    [8.23e-08, 7.99e-08, 7.41e-08, -5.99e-07, -1.87e-06, 1.94e-05, -1.31e-06, 3.54e-06, 5.42e-06, 3.05e-06, -3.68e-08],
    [-7.08e-08, -6.79e-08, -6.30e-08, -2.11e-06, 1.59e-06, -1.31e-06, 9.32e-05, 4.16e-07, 1.36e-05, 1.74e-05,-1.64e-07],
    [9.60e-08, 9.30e-08, 8.62e-08, -3.68e-07, -2.18e-06, 3.54e-06, 4.16e-07, 1.28e-05, 4.96e-06, -4.02e-08, 1.61e-07],
    [6.01e-08, 6.10e-08, 5.65e-08, -3.90e-06, -1.43e-06, 5.42e-06, 1.36e-05, 4.96e-06, 2.79e-05, 2.84e-05, 1.01e-08],
    [3.05e-08, 2.92e-08, 2.70e-08, -4.40e-06, -6.85e-07, 3.05e-06, 1.74e-05, -4.02e-08, 2.84e-05, 3.72e-05, -6.49e-07],
    [-6.13e-09, -5.88e-09, -5.42e-09, 3.93e-08, 1.38e-07, -3.68e-08, -1.64e-07, 1.61e-07, 1.01e-08, -6.49e-07, 5.58e-07]
])

transition_match = {"E1A_3H": ["H3_E", "He3_E"]}

transition_lecs = {"E1A_3H": ["c3", "c4", "c_D"]}

# the position of the lecs being varied (positions are above
# in the nnlo_sat_lecs dictionary and param_names_chiral
transition_lecs_idx = {"E1A_3H": [1, 2, 14]}  # c3, c4, c_D positions

piN_par_names = ["c1", "c3", "c4"]
piN_vals_rs = {
    "c1": -0.7424161689888585,
    "c3": -3.6126853832948798,
    "c4": 2.4390225292517886,
}
piN_sigma_rs = {
    "c1": 0.02471073327924288,
    "c3": 0.05026164230896359,
    "c4": 0.02641360498185206,
}
piN_corr_rs = np.asarray(
    [
        [1.0, 0.5043107492247224, 0.03830003350525791],
        [0.5043107492247224, 1.0, -0.04120164865359903],
        [0.03830003350525791, -0.04120164865359903, 1.0],
    ]
)

piN_cov_mat = np.zeros((3, 3))
for i, lec_i in enumerate(piN_sigma_rs):
    for j, lec_j in enumerate(piN_sigma_rs):
        piN_cov_mat[i, j] = (
            piN_corr_rs[i, j] * piN_sigma_rs[lec_i] * piN_sigma_rs[lec_j]
        )
