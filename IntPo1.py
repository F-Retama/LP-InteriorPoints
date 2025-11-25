# -*- coding: utf-8 -*-
"""
@autores:   Fernando Retama
           Diego Nieto Pizano

           Programación Lineal
                   ITAM
       Método de Puntos Interiores
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


def probAleatorio (m :int, n :int, beta = 0.4):
    A = np.random.rand(m,n)
    c = np.random.rand(n)
    b = np.ones(m)
    # hacer rala la matriz
    for i in range(m):
        for j in range(n):
            if np.abs( A[i,j] ) < beta:
                A[i,j] = 0.0
    return (A, c, b)


def solveIntPo (A, c, b, gamma = 1/2, tol = 12, maxIters = 2000):
    iters = 0
    # valores iniciales
    m, n = A.shape
    x = np.ones(n)
    z = np.ones(n)
    y = np.zeros(m)
    # parte fija del lado izquierdo
    p = n+m+n
    M = np.zeros((p,p))
    M[ 0:n , n:n+m ] = A.T
    M[ 0:n , n+m: ] = np.eye(n)
    M[ n:n+m , 0:n ] = A
    # iteraciones del método de Newton
    err = 1
    while err > 10**-tol and iters < maxIters:
        # parte variable del lado izquierdo para completar F'(w)
        M[ n+m: , 0:n ] = np.diag(z)
        M[ n+m: , -n: ] = np.diag(x)
        # lado derecho para construir -F(w)
        ld_1 = A.T@y + z - c
        ld_2 = A@x - b
        ld_3 = x*z - gamma
        der = -np.hstack((ld_1, ld_2, ld_3))
        # solución del sistema de Newton
        Delta = np.linalg.solve(M, der)
        d_x = Delta[0:n]
        d_y = Delta[n:n+m]
        d_z = Delta[n+m:]
        # recortes para obtener alfa
        alfa_x = np.ones(n)
        alfa_z = np.ones(n)
        for i in range(n):
            if (d_x[i]<0):
                alfa_x[i] = -(x[i]/d_x[i])
            if (d_z[i]<0):
                alfa_z[i] = -(z[i]/d_z[i])
        alfa_xs = np.min(alfa_x)
        alfa_zs = np.min(alfa_z)
        alfa = (0.995)*np.min([alfa_xs,alfa_zs])
        # actualización de la iteración
        err = np.linalg.norm(der, np.inf)
        x = x+alfa*d_x
        y = y+alfa*d_y
        z = z+alfa*d_z
        gamma = x@z/(10*n)
        #gamma = x@z * min( err , 1/(10*n) )
        iters += 1
        #print('gamma', gamma)
        #print('err', err)
    return (x, y, z, iters, err)


afiro = sio.loadmat('israel.mat')
A = afiro['A']
c = afiro['c']
b = afiro['b']
lo = afiro['lo']
hi = afiro['hi']

A_full = A.toarray()
c_full = c.flatten()
b_full = b.flatten()


print(A_full.shape)
print(np.linalg.matrix_rank(A_full))

x, y, z, iters, err = solveIntPo(A_full, c_full, b_full)

print('iters', iters)
print('error', err)
print('valor obtenido', c_full@x)
    
   

