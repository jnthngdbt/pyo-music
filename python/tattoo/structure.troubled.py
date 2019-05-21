# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 23:17:01 2018

@author: jonat
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matplotlib.style.use(['jonat'])

plt.close('all')

G = np.arange(-1,2) # grid
Ng = len(G)
Nz = 8 # number of z levels

GridScale = 1
NoiseScale = 0.01

GetCoordPerturbation = lambda iz: iz*NoiseScale*np.random.randn()

Np = Ng*Ng*Nz # number of points
Pts = np.zeros((3,Np))
i = 0
for iz in np.arange(Nz):
  for iy in np.arange(Ng):
    for ix in np.arange(Ng):
      Pts[0,i] = G[ix]*GridScale + GetCoordPerturbation(iz)
      Pts[1,i] = G[iy]*GridScale + GetCoordPerturbation(iz)
      Pts[2,i] = iz*GridScale + GetCoordPerturbation(iz)
      i += 1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Pts[0],Pts[1],Pts[2])

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.axis('off')

plt.show()