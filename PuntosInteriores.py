"""
@autores:   Fernando Retama
           Diego Nieto Pizano

           Programación Lineal
                   ITAM
       Método de Puntos Interiores
"""

import numpy as np
import scipy.io as sio
from scipy.optimize import linprog
import pandas as pd

# --------------------- Solver Puntos Interiores ------------------------------


def solveIntPo(A, c, b, gamma=1 / 2, tol=10, maxIters=2000):
    iters = 0
    # valores iniciales
    m, n = A.shape
    x = np.ones(n)
    z = np.ones(n)
    y = np.zeros(m)
    # parte fija del lado izquierdo
    p = n + m + n
    M = np.zeros((p, p))
    M[0:n, n : n + m] = A.T
    M[0:n, n + m :] = np.eye(n)
    M[n : n + m, 0:n] = A
    # iteraciones del método de Newton
    err = 1
    while err > 10**-tol and iters < maxIters:
        # parte variable del lado izquierdo para completar F'(w)
        M[n + m :, 0:n] = np.diag(z)
        M[n + m :, -n:] = np.diag(x)
        # lado derecho para construir -F(w)
        ld_1 = A.T @ y + z - c
        ld_2 = A @ x - b
        ld_3 = x * z - gamma
        der = -np.hstack((ld_1, ld_2, ld_3))
        # solución del sistema de Newton
        Delta = np.linalg.solve(M, der)
        d_x = Delta[0:n]
        d_y = Delta[n : n + m]
        d_z = Delta[n + m :]
        # recortes para obtener alfa
        alfa_x = np.ones(n)
        alfa_z = np.ones(n)
        for i in range(n):
            if d_x[i] < 0:
                alfa_x[i] = -(x[i] / d_x[i])
            if d_z[i] < 0:
                alfa_z[i] = -(z[i] / d_z[i])
        alfa_xs = np.min(alfa_x)
        alfa_zs = np.min(alfa_z)
        alfa = (0.995) * np.min([alfa_xs, alfa_zs])
        # actualización de la iteración
        err = np.linalg.norm(der, np.inf)
        x = x + alfa * d_x
        y = y + alfa * d_y
        z = z + alfa * d_z
        gamma = x @ z / (10 * n)
        iters += 1
    return (x, y, z, iters, err)


# ------------------------ Métricas por guardar -------------------------------

problemas = ["agg2", "beaconfd", "brandy", "e226", "israel"]

valor_highs = []
iters_highs = []
conv_highs = []
valor_intPo = []
iters_intPo = []
conv_intPo = []

# ---------------------- Resolución de los problemas --------------------------

for prob in problemas:
    prob += ".mat"
    matlabData = sio.loadmat(prob)
    A_mat = matlabData["A"]
    c_mat = matlabData["c"]
    b_mat = matlabData["b"]
    A = A_mat.toarray()
    c = c_mat.flatten()
    b = b_mat.flatten()
    # Limpieza de datos (para brandy)
    b = b[np.any(A != 0, axis=1)]
    A = A[np.any(A != 0, axis=1)]
    # Solución por Highs
    bounds = [(0, float("inf"))] * (A.shape[1])
    sol_highs = linprog(c, None, None, A, b, bounds, method="highs")
    valor_highs.append(sol_highs.fun)
    iters_highs.append(sol_highs.nit)
    conv_highs.append(sol_highs.success)
    # Solución por Puntos Interiores
    x, y, z, iters, err = solveIntPo(A, c, b)
    val = c @ x
    valor_intPo.append(val)
    iters_intPo.append(iters)
    conv_intPo.append(err < 10**-10)


# --------------------- Exportación de los resultados -------------------------

df = pd.DataFrame(
    {
        "Problema": problemas,
        "Conv Highs": conv_highs,
        "Valor Highs": valor_highs,
        "Iters Highs": iters_highs,
        "Conv punto interior": conv_intPo,
        "Valor punto interior": valor_intPo,
        "Iters punto interior": iters_intPo,
    }
)

df.to_csv('resultados.csv', index=False)
