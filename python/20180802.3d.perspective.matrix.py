# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

matplotlib.style.use(['jonat'])

plt.close('all')

# POINTS: forms a dense cube

Np = 10
p3 = np.array([np.random.rand(3,Np), np.ones((1,Np))]) # homogeneous coordinates

# FRUSTUM

# CAMERA

camposrel = np.array([0,0,0])


# SORT POINTS BY CAMERA DISTANCE

# fig = plt.figure()
# ax = fig.add_subplot(111)

# ax.scatter(Pts[0],Pts[1])

# ax.set_xlabel('x')
# ax.set_ylabel('y')

# plt.axis('off')

plt.show()