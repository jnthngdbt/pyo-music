# I want to find the matrix transform that maps any 8 3d points (not really any, but forming a kind of trapezoid prism) onto a unit cube.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matplotlib.style.use(['dark_background'])
plt.rcParams['grid.color'] = '#333333'
plt.rcParams['figure.facecolor'] = '#222222'
plt.rcParams['axes.facecolor'] = '#222222'

plt.close('all')

units = 1

# -----------------------------------------------------------------
M = np.array([
  [-1,  1, -1], # back-bottom-left
  [-1,  1,  1], # back-top-left
  [ 1,  1,  1], # back-top-right
  [ 1,  1, -1], # back-bottom-right
  [-1, -1, -1], # front-bottom-left
  [-1, -1,  1], # front-top-left
  [ 1, -1,  1], # front-top-right
  [ 1, -1, -1], # front-bottom-right
])

M = M.T
M *= units

# --------------------------------------------------------
def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    '''
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)

# -----------------------------------------------------------------
def getAxe2d():
  fig = plt.figure()
  ax = fig.add_subplot(111)
  return fig, ax

def getAxe3d():
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  return fig, ax

def getAxe(type):
  if (type == '2d'): return getAxe2d()
  if (type == '3d'): return getAxe3d()

def setAxe2d(fig, ax):
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  fig.tight_layout()
  plt.axis('square')

def setAxe3d(fig, ax):
  setAxe2d(fig, ax)
  ax.set_zlabel('Z')
  ax.w_xaxis.set_pane_color((0,0,0,0))
  ax.w_yaxis.set_pane_color((0,0,0,0))
  ax.w_zaxis.set_pane_color((0,0,0,0))
  set_axes_equal(ax)

def setAxe(fig, ax, type):
  if (type == '2d'): return setAxe2d(fig, ax)
  if (type == '3d'): return setAxe3d(fig, ax)

# -----------------------------------------------------------------
def draw2d(ax, pts, idx, opt):
  ax.plot(pts[0,idx], pts[1,idx], opt, color='grey', markersize=10)

def draw3d(ax, pts, idx, opt):
  ax.plot(pts[0,idx], pts[1,idx], pts[2,idx], opt, color='grey', markersize=10)

def draw(ax, pts, type, idx):
  opt = '.' if len(idx) == 1 else '-'
  if (type == '2d'): draw2d(ax, pts, idx, opt)
  if (type == '3d'): draw3d(ax, pts, idx, opt)

# -----------------------------------------------------------------
def drawCube(ax, pts, type):
  draw(ax, pts, type, [0,1,2,3,0])
  draw(ax, pts, type, [4,5,6,7,4])
  draw(ax, pts, type, [1,5])
  draw(ax, pts, type, [2,6])
  draw(ax, pts, type, [0,4])
  draw(ax, pts, type, [3,7])
  draw(ax, pts, type, [0])

def plotCube2d(pts):
  fig, ax = getAxe2d()
  drawCube(ax, pts, '2d')
  setAxe2d(fig, ax)
  return ax

def plotCube3d(pts):
  fig, ax = getAxe3d()
  drawCube(ax, pts, '3d')
  setAxe3d(fig, ax)
  return ax

def normalizeHomogeneousPoints(p):
  q = np.zeros(p.shape)
  for i, v in enumerate(p.T):
    q[:,i] = v / v[3]
  return q

# -----------------------------------------------------------------
lim = 1.5

ax = plotCube3d(M)
plt.title('Cube M')
plt.xlim([-lim,lim])
plt.ylim([-lim,lim])
ax.set_zlim([-lim,lim])

# -----------------------------------------------------------------
MI = np.linalg.pinv(M)

ax = plotCube3d(MI.T)
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

ax = plotCube3d(N)
plt.title('Cube warped N')
plt.xlim([-lim,lim])
plt.ylim([-lim,lim])
ax.set_zlim([-lim,lim])

NI = np.linalg.pinv(N)

ax = plotCube3d(NI.T)
plt.title('Cube warped inverse NI.T')
plt.xlim([-lim,lim])
plt.ylim([-lim,lim])
ax.set_zlim([-lim,lim])

T = np.dot(M, NI)

Me = np.dot(T, N)

ax = plotCube3d(Me)
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
Mhe = normalizeHomogeneousPoints(MHe)

ax = plotCube3d(Mhe)
plt.title('Cube warped mapped to cube (homo) Mhe')
plt.xlim([-lim,lim])
plt.ylim([-lim,lim])
ax.set_zlim([-lim,lim])

# -----------------------------------------------------------------
plt.show()