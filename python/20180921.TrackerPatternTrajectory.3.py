# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

# --------------------------------------------------------
# PASTE THE DEBUG DATA HERE
# --------------------------------------------------------

patternNames = [
  'P1L',
  'P1R',
  'P2R'
]

patternPoints = [ # RT, AX, AY
  [[ 309.685,	858.591, 342.532], [ 244.684,	818.526, 342.95 ], [ 310.618,	859.06 , 220.297]],
  [[  99.803,	688.955,  27.298], [  55.766,	661.937,  27.421], [  99.861,	688.749, 117.033]],
  [[  69.845,	712.822, 342.185], [ 135.111,	752.058, 342.626], [  70.038,	711.763, 219.551]]
]

patternHits = [
  [215., 800., 201.],
  [230., 768., 204.],
  [213., 797., 200.]
]

position = [500, 0, 150]
direction = [0, 1, 0]

# -----------------------------------------------------------------------------
patterns = []

def makePattern(corners):
  c = np.zeros((4,3))
  c[0:3,:] = corners
  c[3,:] = c[0] + (c[1]-c[0]) + (c[2]-c[0]) # AXY
  return c

for p in patternPoints:
  patterns.append(makePattern(p))

numPatterns = len(patterns)

# --------------------------------------------------------

position = np.array(position, ndmin=2).T
direction = np.array(direction, ndmin=2).T

# --------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig.tight_layout()

# --------------------------------------------------------
def drawPatternRectangle(idx):
  p = patterns[idx]
  li = [0,1,3,2,0] # loop corners
  ax.plot(p[li,0], p[li,1], p[li,2], 'C'+str(idx)) # rectangle lines

def drawPatternRoot(idx):
  p = patterns[idx]
  x = p[0,0]
  y = p[0,1]
  z = p[0,2]
  ax.plot([x], [y], [z], '.', color='C'+str(idx)) # root point
  ax.text(x, y, z, patternNames[idx])

def drawPatternHit(idx):
  h = patternHits[idx]
  r = patterns[idx][0] # root
  ax.scatter(h[0], h[1], h[2], 'C'+str(idx)) # hit point
  ax.plot([r[0],h[0]], [r[1],h[1]], [r[2],h[2]], ':', color='C'+str(idx)) # line from root to hit

for i in np.arange(numPatterns):
  drawPatternRectangle(i)
  drawPatternRoot(i)
  drawPatternHit(i)

# --------------------------------------------------------
def drawTrajectory(pos, dir):
  p1 = pos[:,0]
  p2 = p1 + 1000*dir[:,0]
  ax.scatter([p1[0]], [p1[1]], [p1[2]], color='C'+str(numPatterns)) # point
  ax.plot([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]], color='C'+str(numPatterns)) # line

drawTrajectory(position, direction)

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

plt.axis('square')
set_axes_equal(ax)

# --------------------------------------------------------
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.w_xaxis.set_pane_color((0,0,0,0))
ax.w_yaxis.set_pane_color((0,0,0,0))
ax.w_zaxis.set_pane_color((0,0,0,0))

plt.show()