# I want to find the matrix transform that maps any 8 3d points (not really any, but forming a kind of trapezoid prism) onto a unit cube.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from library import m3d

# -----------------------------------------------------------------
lim = 1.5

M = m3d.makeCube(1)

ax = m3d.plotCube3d(M)
plt.title('Cube M')
plt.xlim([-lim,lim])
plt.ylim([-lim,lim])
ax.set_zlim([-lim,lim])

# -----------------------------------------------------------------
MI = np.linalg.pinv(M)

ax = m3d.plotCube3d(MI.T)
plt.title('Cube inverse MI.T')
plt.xlim([-lim,lim])
plt.ylim([-lim,lim])
ax.set_zlim([-lim,lim])

# -----------------------------------------------------------------
N = np.zeros(M.shape)
for i in np.arange(8):
  N[:,i] = 0.1 * np.random.randn() + M[:,i]
# R = np.random.rand() * np.eye(3)
# N = np.dot(R, M)

ax = m3d.plotCube3d(N)
plt.title('Cube warped N')
plt.xlim([-lim,lim])
plt.ylim([-lim,lim])
ax.set_zlim([-lim,lim])

NI = np.linalg.pinv(N)

ax = m3d.plotCube3d(NI.T)
plt.title('Cube warped inverse NI.T')
plt.xlim([-lim,lim])
plt.ylim([-lim,lim])
ax.set_zlim([-lim,lim])

T = np.dot(M, NI)

Me = np.dot(T, N)

ax = m3d.plotCube3d(Me)
plt.title('Cube warped mapped to cube Me')
plt.xlim([-lim,lim])
plt.ylim([-lim,lim])
ax.set_zlim([-lim,lim])

# -----------------------------------------------------------------
# Using homo

MH = np.vstack([M,  np.ones((1,8))])
NH = np.vstack([N,  np.ones((1,8))])
NHI = np.linalg.pinv(NH)
TH = np.dot(MH, NHI)
MHe = np.dot(TH, NH)
Mhe = m3d.normalizeHomogeneousPoints(MHe)

ax = m3d.plotCube3d(Mhe)
plt.title('Cube warped mapped to cube (homo) Mhe')
plt.xlim([-lim,lim])
plt.ylim([-lim,lim])
ax.set_zlim([-lim,lim])

# -----------------------------------------------------------------
plt.show()