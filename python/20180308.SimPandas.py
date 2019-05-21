# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 20:06:27 2018

@author: jonat
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('jonat')

d = {
  'time': np.arange(0.0, 1.0, 1.0/1000),
  'measure': np.random.rand(10,1000)
}

filt = np.ones((1,50))

for m in d['measure']: 
  m = np.convolve(m.flatten(),filt.flatten(),'same')
  
plt.plot(d['time'], d['measure'].T)

plt.show()