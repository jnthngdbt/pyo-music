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

def generateClusters():
  N = 4000
  c = [[2,2],
       [10,2],
       [4,5]]
  x = []
  y = []
  for ci in c:
    x = np.concatenate((x, np.random.randn(N) + ci[0]));
    y = np.concatenate((y, np.random.randn(N) + ci[1]));
  return x, y
  
def generateWall():
  N = 5000
  x = np.random.randn(N);
  y = np.random.random(N);
  return x, y

# -
  
#x, y = generateWall()
x, y = generateClusters()

# -

fig1 = plt.figure(
    figsize=(10,10), 
) #
ax1 = fig1.add_subplot(111)

ax1.plot(x, y,
         '.', 
         color=(0.6,0.6,0.6), 
         alpha=0.5,
         markersize=2,
         linewidth=0.1
)

# -

H, xe, ye = np.histogram2d(x,y,10)
HT = H.T
HTF = HT.flatten()
Hn = (HT-np.min(HTF)) / (np.max(HTF[:])-np.min(HTF[:]))
C = cm(Hn)
#C[:,:,3] = np.sqrt(Hn)

#ax2 = fig1.add_subplot(122)
ax2 = ax1
ax2.imshow(C, 
           cmap=cm,
           alpha=0.8,
           origin='lower',
           interpolation='bicubic',
           extent=[xe[0], xe[-1], ye[0], ye[-1]])

plt.axis('off')

plt.show()