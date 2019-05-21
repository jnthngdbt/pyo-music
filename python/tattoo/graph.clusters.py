# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 23:17:01 2018

@author: jonat
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use(['jonat'])

plt.close('all')
cm = matplotlib.colors.LinearSegmentedColormap.from_list(
    'mycm',
    [(.1,.1,.1), (0.8,0.6,1.0)],
    N=1024)

Nc = 100
Ng = 3
N = Ng*Nc

clr = (0.2,0.2,0.2)

def generateClusters():
  sc = 0.9
  sp = 1.5
  c = [[0,0],
       [6,10],
       [10,-2]]
  x = []
  y = []
  for ci in c:
    x = np.concatenate((x, sp*np.random.randn(Nc) + sc*ci[0]))
    y = np.concatenate((y, sp*np.random.randn(Nc) + sc*ci[1]))
  return x, y

def generateEdges(x,y):
  for i in np.arange(N):
    for j in np.arange(i+1,N):
      xi = x[i]
      yi = y[i]
      xj = x[j]
      yj = y[j]

      dx = xi-xj
      dy = yi-yj
      d = np.sqrt(dx**2 + dy**2)

      if (d < 8):
        plt.plot([xi,xj], [yi,yj],
          color=clr, 
         alpha=0.5,
         markersize=2,
         linewidth=0.1
        )

# -
  
x, y = generateClusters()

# -

fig1 = plt.figure(
    figsize=(10,10), 
) #
ax1 = fig1.add_subplot(111)

ax1.plot(x, y,
         '.', 
         color=clr, 
         alpha=0.5,
         markersize=5,
         linewidth=0.1
)
fig1.patch.set_facecolor((0.8,0.8,0.8))

generateEdges(x,y)

plt.axis('off')

plt.show()