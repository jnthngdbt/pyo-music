import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use(['dark_background'])

print("Creating data ({},{})...".format(3,100))
a = np.random.randn(3,100)

plt.figure()
plt.plot(a.T)
plt.show()