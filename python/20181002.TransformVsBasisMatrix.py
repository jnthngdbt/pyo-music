import numpy as np
import matplotlib.pyplot as plt

plt.style.use('dark_background')
plt.rcParams['#grid.color'] = '#333333'
plt.rcParams['figure.facecolor'] = '#222222'
plt.rcParams['axes.facecolor'] = '#222222'

plt.close('all')

# -----------------------------------------------------------------
p = np.array([
  [ 0, 0, 0],
  [ 2, 0, 0],
  [ 1, 1, 0],
])

N = p.shape[0]
p = p.T
p = np.vstack([p,  np.ones((1,N))])