import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use(['dark_background'])
plt.rcParams['grid.color'] = '#333333'
plt.rcParams['figure.facecolor'] = '#222222'
plt.rcParams['axes.facecolor'] = '#222222'

plt.close('all')

tolLabels = ['0.010', '0.020', '0.030']
for tolLabel in tolLabels:
    tol = float(tolLabel)
    rms = np.arange(0.00001, 0.05, 0.0001)
    score = np.exp(-rms/tol)
    plt.plot(rms, score)

plt.xlabel('rms')
plt.ylabel('score')
plt.title('effect of normalization distance')
plt.legend(tolLabels)

plt.show()