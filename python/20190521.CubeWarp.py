# I want to find the matrix transform that maps any 8 3d points (not really any, but forming a kind of trapezoid prism) onto a unit cube.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from library import m3d

# -----------------------------------------------------------------
lim = 1.5

M = m3d.makeCube()

ax = m3d.plotCube3d(M)
plt.title('Cube M')
plt.xlim([-lim,lim])
plt.ylim([-lim,lim])
ax.set_zlim([-lim,lim])

# -----------------------------------------------------------------
MI = np.linalg.pinv(M)

# -----------------------------------------------------------------
N = np.array([
  [-1.0,  1.0, -1.0], # left-back-bottom
  [-1.0,  0.7,  1.0], # left-back-top
  [ 1.0,  0.7,  1.0], # right-back-top
  [ 1.0,  1.0, -1.0], # right-back-bottom
  [-1.0, -1.0, -1.0], # left-front-bottom
  [-1.0, -0.4,  0.8], # left-front-top
  [ 1.0, -0.4,  0.8], # right-front-top
  [ 1.0, -1.0, -1.0], # right-front-bottom
])

N = N.T

ax = m3d.plotCube3d(N)
plt.title('Cube warped N')
plt.xlim([-lim,lim])
plt.ylim([-lim,lim])
ax.set_zlim([-lim,lim])

NI = np.linalg.pinv(N)

T = np.dot(M, NI)

Me = np.dot(T, N)

ax = m3d.plotCube3d(Me)
plt.title('Cube warped mapped to cube Me')
plt.xlim([-lim,lim])
plt.ylim([-lim,lim])
ax.set_zlim([-lim,lim])

print(N.T)
print(T)
print(Me.T)

# -----------------------------------------------------------------
# NOTE: NOT USEFUL TO USE 4D HOMOGENEOUS COORDINATES. IT WILL GIVE
# SAME RESULTS AS 3D, SINCE THE CUBE POINTS LIE IN A HYPERPLANE IN 
# 4D SPACE.

# -----------------------------------------------------------------
plt.show()