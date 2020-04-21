import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use(['dark_background'])
plt.rcParams['grid.color'] = '#333333'
plt.rcParams['figure.facecolor'] = '#222222'
plt.rcParams['axes.facecolor'] = '#222222'

plt.close('all')

# < ------- W -------- >
#  __    __    __    __    ^
# |  |  |  |  |  |  |  |   | H
# |  |  |  |  |  |  |  |   | 
# |  |__|  |__|  |__|  |   v
#
# N = 8, W = 14, H = 3 ====> W + H*N

#  __    __    __    
# |  |  |  |  |  |  |
# |  |  |  |  |  |  |
# |  |__|  |__|  |__|
#
# N = 7, W = 12, H = 3

units = '\''
H = [4]   # height of the light rectangle
W = [4, 8]   # width of the light rectangle
N = 5   # number of vertical lines
C = [30, 50]  # length of available light cable, N = floor[(C-W)/H]

xlim = [0, np.max(W)]
ylim = [0, np.max(H)]

xOff = 0.1 * (xlim[1] - xlim[0])
yOff = 0.1 * (ylim[1] - ylim[0])

xlim = [xlim[0]-xOff, xlim[1]+xOff]
ylim = [ylim[0]-yOff, ylim[1]+yOff]

# =======================================================

def plotNeededCableLenght(H,W):
    n = np.arange(1, 15)
    L = W + H*n

    plt.figure()

    plt.plot(n,L)

    plt.xlabel('number of vertical lines')
    plt.ylabel('light cable length ({})'.format(units))
    plt.title('light rectangle H X W ({}{} X {}{})'.format(H, units, W, units))
    plt.xlim(xlim)
    plt.ylim(ylim)

for Hi in H:
    for Wi in W:
        plotNeededCableLenght(Hi,Wi)

# =======================================================
# Draw get specs

def drawLightRectangle(C, H, W):
    Nc = np.floor((C-W)/H) # number of vertical lines for given constraints
    d = W/(Nc-1) # distance between vertical lines

    x = []
    y = []
    ny = 1 # 
    for i in np.arange(Nc):
        x.append(i*d)
        x.append(i*d)
        if ny > 0:
            y.append(0)
            y.append(H)
        else:
            y.append(H)
            y.append(0)
        ny *= -1

    plt.figure()

    plt.plot(x,y)

    plt.xlabel('width ({})'.format(units))
    plt.ylabel('height ({})'.format(units))
    plt.title('light rectangle for light cable of length {}{}'.format(C, units))
    plt.xlim(xlim)
    plt.ylim(ylim)

for Hi in H:
    for Wi in W:
        for Ci in C:
            drawLightRectangle(Ci, Hi, Wi)

# =======================================================

plt.show()