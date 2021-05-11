import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use(['dark_background'])
plt.rcParams['grid.color'] = '#333333'
plt.rcParams['figure.facecolor'] = '#222222'
plt.rcParams['axes.facecolor'] = '#222222'

plt.close('all')

Nt = 365
Np = 2 # number of predictors
Ne = 5 # number of events

def cumsumStock():
  x = np.random.normal(0, 1, Nt)
  x = np.cumsum(x)
  x = x + 0.2 * np.arange(0, Nt)
  return x

while(1):

  s = cumsumStock() # stocks baseline

  i = np.random.randint(0.1 * Nt, 0.9 * Nt, Ne)

  p = np.zeros((Np, Nt))
  for k in i:
    p[:, k] = np.random.normal(size=2)
    
  for k in np.arange(Np):
    p[k, :] = np.convolve(p[k, :], np.ones(3), 'same')

  s = np.random.normal(0, 1, Nt)
  s = s + 10 * p[0, :] + 10 * p[1, :]
  s = np.cumsum(s)
  s = s + 0.2 * np.arange(0, Nt)

  plt.figure(figsize= 0.005 * np.array([1334, 750])) # iPhone 6, 7, 8 resolution

  plt.subplot(211)
  plt.plot(p.T)
  plt.axis('off')

  plt.subplot(212)
  plt.plot(s)
  plt.axis('off')

  plt.tight_layout()
  plt.show()
  plt.close('all')