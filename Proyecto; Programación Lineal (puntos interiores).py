#!/usr/bin/env python
# coding: utf-8

# In[12]:





# In[ ]:





# In[41]:


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


def solveIntPo (A, c, b, gamma = 1/2, tol = 10, maxIters = 2000):
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
        iters += 1
    return (x, y, z, iters, err)



#-------------------------------- 1. AGG2 -------------------------------------

agg2 = sio.loadmat('agg2.mat')
A = agg2['A']
c = agg2['c']
b = agg2['b']

A1 = A.toarray()
c1 = c.flatten()
b1 = b.flatten()

bounds = [(0,float('inf'))]*(A1.shape[1])
agg2_highs = linprog( c1, None, None, A1, b1, bounds, method='highs')

#----------------------------- 2. BEACONFD ------------------------------------

beaconfd = sio.loadmat('beaconfd.mat')
A = beaconfd['A']
c = beaconfd['c']
b = beaconfd['b']

A2 = A.toarray()
c2 = c.flatten()
b2 = b.flatten()

bounds = [(0,float('inf'))]*(A2.shape[1])
beaconfd_highs = linprog( c2, None, None, A2, b2, bounds, method='highs')

#------------------------------ 3. BRANDY -------------------------------------

brandy = sio.loadmat('brandy.mat')
A = brandy['A']
c = brandy['c']
b = brandy['b']

A3 = A.toarray()
c3 = c.flatten()
b3 = b.flatten()
b3 = b3[ np.any(A3 != 0 , axis = 1) ]
A3 = A3[ np.any(A3 != 0 , axis = 1) ]

bounds = [(0,float('inf'))]*(A3.shape[1])
brandy_highs = linprog( c3, None, None, A3, b3, bounds, method='highs')

#-------------------------------- 4. E226 -------------------------------------

e226 = sio.loadmat('e226.mat')
A = e226['A']
c = e226['c']
b = e226['b']

A4 = A.toarray()
c4 = c.flatten()
b4 = b.flatten()

bounds = [(0,float('inf'))]*(A4.shape[1])
e226_highs = linprog( c4, None, None, A4, b4, bounds, method='highs')

#------------------------------- 5. Israel ------------------------------------

israel = sio.loadmat('israel.mat')
A = israel['A']
c = israel['c']
b = israel['b']

A5 = A.toarray()
c5 = c.flatten()
b5 = b.flatten()

bounds = [(0,float('inf'))]*(A5.shape[1])
israel_highs = linprog( c5, None, None, A5, b5, bounds, method='highs')

# listas finales
highs_valor=[agg2_highs.fun,beaconfd_highs.fun,e226_highs.fun,israel_highs.fun,brandy_highs.fun]
highs_iteraciones=[agg2_highs.nit,beaconfd_highs.nit,e226_highs.nit,israel_highs.nit,brandy_highs.nit]

#------------------------------- 6. con puntos interiores ----------------------
ejemplos=["agg2.mat","beaconfd.mat","e226.mat","israel.mat"]
iteraciones=[]
error=[]
valor_obtenido=[]


for i in range(len(ejemplos)):
    afiro = sio.loadmat(ejemplos[i])
    A = afiro['A']
    c = afiro['c']
    b = afiro['b']
    lo = afiro['lo']
    hi = afiro['hi']

    A_full = A.toarray()
    c_full = c.flatten()
    b_full = b.flatten()


    x, y, z, iters, err = solveIntPo(A_full, c_full, b_full)
    valor=c_full@x
    
    iteraciones.append(iters)
    error.append(err)
    valor_obtenido.append(valor)

# caso de brandy
x3, y3, z3, iters3, err3 = solveIntPo(A3, c3, b3)
valor3=c3@x3

iteraciones.append(iters3)
error.append(err3)
valor_obtenido.append(valor3)

#---------------------------- data frame -------------------------------
algoritmos=["agg2","beaconfd","e226","israel","brandy"]
df = pd.DataFrame({
    "Algoritmo": algoritmos,
    "Valor HIGHS": highs_valor,
    "Iter HIGHS": highs_iteraciones,
    "Valor punto interior":valor_obtenido,
    "Iter punto interior":iteraciones,
    "Error":error
})

df


# In[ ]:




