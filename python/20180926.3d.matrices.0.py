# -*- coding: utf-8 -*-

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
modelPoints = np.array([
  [-1,  1, -1], # back-bottom-left
  [-1,  1,  1], # back-top-left
  [ 1,  1,  1], # back-top-right
  [ 1,  1, -1], # back-bottom-right
  [-1, -1, -1], # front-bottom-left
  [-1, -1,  1], # front-top-left
  [ 1, -1,  1], # front-top-right
  [ 1, -1, -1], # front-bottom-right
])

modelPoints = modelPoints.T
modelPoints *= units
modelPoints = np.vstack([modelPoints,  np.ones((1,8))])

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

def plotCube3d(pts):
  fig, ax = getAxe3d()
  drawCube(ax, pts, '3d')
  setAxe3d(fig, ax)

# -----------------------------------------------------------------
def normalizeVector(v):
  vf = v.flatten()
  return vf / np.sqrt(np.dot(v,v))

def normalizeMatrix(M):
  for v in M:
    v = normalizeVector(v)
  return M

# -----------------------------------------------------------------
def getBasisPoints():
  return np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [1,1,1],
  ])

def drawBasis(ax, M, type, matrixType):
  axeColors = ['r','g','b']
  B = invertHomogeneousMatrix(M) if matrixType == 'transform' else M
  trans = B[:,3] if B.shape[1] >= 4 else np.zeros(4)
  def line(i, v, trans): return [trans[i], trans[i]+v[i]]
  for i, v in enumerate(B.T): # transpose such that each row is an axis of the basis
    if (i < 3):
      if (type == '2d'): ax.plot(line(0,v,trans), line(1,v,trans), axeColors[i])
      if (type == '3d'): ax.plot(line(0,v,trans), line(1,v,trans), line(2,v,trans), axeColors[i])

# -----------------------------------------------------------------
def invertHomogeneousMatrix(M):
  R = M[:3,:3]
  t = M[:3,3]
  N = np.eye(4)
  N[:3,:3] = R.T
  N[:3,3] = np.dot(R.T, t)
  return N

def createTransformMatrix(lookat, translation, type):
  # Orthonormal basis used to define the rotation matrix
  Mz = normalizeVector(-lookat)
  Mx = normalizeVector(np.cross(np.array([0,1,0]), Mz))
  My = normalizeVector(np.cross(Mz, Mx))

  MT = translation

  M = np.array([
    [Mx[0], My[0], Mz[0], MT[0]],
    [Mx[1], My[1], Mz[1], MT[1]],
    [Mx[2], My[2], Mz[2], MT[2]],
    [    0,     0,     0,     1],
  ])

  # We want the transform whose basis has the pose given as arguments.
  if (type == 'basis'):
    return M

  # We want the transform such that transforming an identity matrix
  # will put it in the pose given as arguments.
  if (type == 'transform'):
    return invertHomogeneousMatrix(M)

  return []

# -----------------------------------------------------------------
modelToWorld = createTransformMatrix(np.array([0.5, 0.3, 0.4]), np.array([1.5, 3.5, 2.5]) * units, 'transform')
worldPoints = np.dot(modelToWorld, modelPoints)

# -----------------------------------------------------------------
def plotTransform(A, Ap, B, Bp, type):
  fig, ax = getAxe(type)
  drawBasis(ax, A, type, 'transform')
  drawBasis(ax, B, type, 'transform')
  drawCube(ax, Ap, type)
  drawCube(ax, Bp, type)
  setAxe(fig, ax, type)

O = np.eye(4)
plotTransform(O, modelPoints, modelToWorld, worldPoints, '3d')
plt.title('3d world matrix')
plotTransform(O, modelPoints, modelToWorld, worldPoints, '2d')
plt.title('2d world matrix')

# -----------------------------------------------------------------
cam = createTransformMatrix(np.array([0.8, 0.3, 0.3]), np.array([-5.0, 1.0, -1.0]) * units, 'basis')

type = '3d'
fig, ax = getAxe(type)
drawBasis(ax, O, type, 'basis')
drawBasis(ax, cam, type, 'basis')
drawCube(ax, worldPoints, type)
setAxe(fig, ax, type)

# -----------------------------------------------------------------

plt.show()