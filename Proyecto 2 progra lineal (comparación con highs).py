#!/usr/bin/env python
# coding: utf-8

# ### Leer archivos .mat en python

# Los problemas son de la forma:
# Min    c'*x
# s.a.   Ax=b
#        x>=0
#       
# Extraer los datos:
#     matriz A de mxn
#     vector c de dimensión n
#     vector b de dimensión m
# ......................................................................................

# In[19]:


import scipy.io as sio 
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import numpy as np 



#----------------------------------------------------------------------------------------------------------------
agg2=sio.loadmat('agg2.mat')
A1=agg2['A']
c1=agg2['c']
b1=agg2['b']
lo1=agg2['lo']
hi1=agg2['hi']

A1_full=A1.toarray()
#b_full=b.toarray()
#c_full=c.toarray()
#n=len(c_full)
#m=len(b_full)
n1=len(c1)
m1=len(b1)

bounds1=[]
w1=(0,float('inf'))
for i in range(n1):
    bounds1.append(w1)

res_agg2=linprog(c1,None,None, A1_full,b1,bounds1,method='highs')
#----------------------------------------------------------------------------------------------------------------
beaconfd=sio.loadmat('beaconfd.mat')
A2=beaconfd['A']
c2=beaconfd['c']
b2=beaconfd['b']
lo2=beaconfd['lo']
hi2=beaconfd['hi']

A2_full=A2.toarray()
#b2_full=b2.toarray()
#c2_full=c2.toarray()
#n=len(c_full)
#m=len(b_full)
n2=len(c2)
m2=len(b2)

bounds2=[]
w2=(0,float('inf'))
for i in range(n2):
    bounds2.append(w2)

res_beaconfd=linprog(c2,None,None, A2_full,b2,bounds2,method='highs')
#----------------------------------------------------------------------------------------------------------------
brandy=sio.loadmat('brandy.mat')
A3=brandy['A']
c3=brandy['c']
b3=brandy['b']
lo3=brandy['lo']
hi3=brandy['hi']

A3_full=A3.toarray()
#b2_full=b2.toarray()
#c2_full=c2.toarray()
#n=len(c_full)
#m=len(b_full)
n3=len(c3)
m3=len(b3)

bounds3=[]
w3=(0,float('inf'))
for i in range(n3):
    bounds3.append(w3)

res_brandy=linprog(c3,None,None, A3_full,b3,bounds3,method='highs')
#----------------------------------------------------------------------------------------------------------------
e226=sio.loadmat('e226.mat')
A4=e226['A']
c4=e226['c']
b4=e226['b']
lo4=e226['lo']
hi4=e226['hi']

A4_full=A4.toarray()
#b2_full=b2.toarray()
#c2_full=c2.toarray()
#n=len(c_full)
#m=len(b_full)
n4=len(c4)
m4=len(b4)

bounds4=[]
w4=(0,float('inf'))
for i in range(n4):
    bounds4.append(w4)

res_e226=linprog(c4,None,None, A4_full,b4,bounds4,method='highs')
#----------------------------------------------------------------------------------------------------------------
israel=sio.loadmat('israel.mat')
A5=israel['A']
c5=israel['c']
b5=israel['b']
lo5=israel['lo']
hi5=israel['hi']

A5_full=A5.toarray()
#b2_full=b2.toarray()
#c2_full=c2.toarray()
#n=len(c_full)
#m=len(b_full)
n5=len(c5)
m5=len(b5)

bounds5=[]
w5=(0,float('inf'))
for i in range(n5):
    bounds5.append(w5)

res_israel=linprog(c5,None,None, A5_full,b5,bounds5,method='highs')
#----------------------------------------------------------------------------------------------------------------


print("Para tener un punto de comparación de nuestro método de puntos interiores podemos tomar en cuenta los siguientes resultados con linprog y 'highs' de python:\n")
print("\nagg2:")
print("solución óptima: ",res_agg2.fun)
print("número de iteraciones: ",res_agg2.nit)
print("\nbeaconfd:")
print("solución óptima: ",res_beaconfd.fun)
print("número de iteraciones: ",res_beaconfd.nit)
print("\nbrandy:")
print("solución óptima: ",res_brandy.fun)
print("número de iteraciones: ",res_brandy.nit)
print("\ne226:")
print("solución óptima: ",res_e226.fun)
print("número de iteraciones: ",res_e226.nit)
print("\nisrael:")
print("solución óptima: ",res_israel.fun)
print("número de iteraciones: ",res_israel.nit)


# In[ ]:





# In[ ]:




