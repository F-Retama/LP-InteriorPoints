# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 00:05:24 2025

@author: retam
"""

import scipy.io as sio

brandy = sio.loadmat('brandy.mat')
A = brandy['A']
c = brandy['c']
b = brandy['b']
lo = brandy['lo']
hi = brandy['hi']

A_full = A.toarray()
c_full = c.flatten()
b_full = b.flatten()

m, n = A_full.shape

rens_nulos = []

for i in range(m):
    flag = True
    for j in range(n):
        if A_full[i][j] != 0.0: flag = False
    if flag: 
        # marcar renglones de ceros
        rens_nulos.append(i)
        if b[i] == 0:
            # marcar si también hay cero en b
            rens_nulos[-1] += 0.5
        
print(rens_nulos)