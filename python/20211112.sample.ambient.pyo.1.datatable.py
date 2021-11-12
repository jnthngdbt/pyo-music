from pyo import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

fs = 44100
T = 4.0

s = Server(sr=fs).boot()

N = int(T * fs)
t = T * np.arange(N)/N
fx = 400.0
x = np.sin(2.0*np.pi*fx*t)

tab = DataTable(size=N, init=x.tolist())

a = Osc(table=tab, freq=tab.getRate(), mul=.4,).out()

s.gui(locals())

plt.plot(t, x)

plt.show()