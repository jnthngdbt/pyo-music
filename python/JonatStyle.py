# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 14:40:06 2018

@author: jonat
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('jonat')

plt.close('all')

x = np.random.randn(300,3)
y = np.random.randn(300,3)

plt.plot(x,y,'.')
plt.title('How \'bout that style')
plt.xlabel('some variable')
plt.ylabel('effect of that variable')
plt.show()