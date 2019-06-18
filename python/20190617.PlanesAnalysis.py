import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd

matplotlib.style.use(['dark_background'])
plt.rcParams['grid.color'] = '#333333'
plt.rcParams['figure.facecolor'] = '#222222'
plt.rcParams['axes.facecolor'] = '#222222'
plt.rcParams["font.size"] = "10"
plt.rcParams["font.family"] = "consolas"

plt.close('all')

# -----------------------------------------------------------------------
# Some feature subsets. Select one to use.

featureSubset_ABCD = [ 'mold', 'backA', 'backB', 'backC', 'backD', 'frontA', 'frontB', 'frontC', 'frontD', 'bottomA', 'bottomB', 'bottomC', 'bottomD', 'topA', 'topB', 'topC', 'topD', 'rightA', 'rightB', 'rightC', 'rightD', 'leftA', 'leftB', 'leftC', 'leftD']
featureSubset_ABCDK = [ 'mold', 'backK', 'frontK', 'bottomK', 'topK', 'rightK', 'leftK', 'backA', 'backB', 'backC', 'backD', 'frontA', 'frontB', 'frontC', 'frontD', 'bottomA', 'bottomB', 'bottomC', 'bottomD', 'topA', 'topB', 'topC', 'topD', 'rightA', 'rightB', 'rightC', 'rightD', 'leftA', 'leftB', 'leftC', 'leftD']

# Selection
featureSubset = featureSubset_ABCDK

# -----------------------------------------------------------------------

scanDataFull = pd.read_csv("python/data/20190617.planes.scans.csv", index_col=False)
moldDataFull = pd.read_csv('python/data/20190617.planes.molds.csv', index_col=False)
moldScanDataFull = pd.read_csv('python/data/20190617.planes.moldscans.csv', index_col=False)

print(scanDataFull)
print(moldDataFull)
print(moldScanDataFull)

def removeNanRows(df):
    nanRows = df.isna().any(axis=1)

    print('removing following NaN rows:')
    print(df[nanRows == True])

    return df[nanRows != True]

scanDataFull = removeNanRows(scanDataFull)
moldDataFull = removeNanRows(moldDataFull)
moldScanDataFull = removeNanRows(moldScanDataFull)

scanData = scanDataFull.loc[:, featureSubset]
moldData = moldDataFull.loc[:, featureSubset]
moldScanData = moldScanDataFull.loc[:, featureSubset]

# scanData.hist(bins=50)
# plt.suptitle('scan data')

# moldData.hist(bins=50)
# plt.suptitle('mold data')

# moldScanData.hist(bins=50)
# plt.suptitle('mold scan data')

# plt.show()

# -----------------------------------------------------------------------

scanData['type'] = 'scan'
moldData['type'] = 'mold'
moldScanData['type'] = 'moldscan'

print('added data type column')
print(scanData)
print(moldData)
print(moldScanData)

data = moldData ############################## scandata
data = data.append(moldScanData, ignore_index=True)

print('appended into a single dataframe')
print(data)

# data.groupby('type').mean().plot.bar()

# plt.show()

# -----------------------------------------------------------------------

def getMoldRowIdx(moldIdx):
    i = data.index[(data['type'] == 'mold') & (data['mold'] == moldIdx)].values
    if len(i) > 0:
        return i[0]
    else:
        return np.nan

data['moldRow'] = [getMoldRowIdx(moldIdx) for moldIdx in list(data['mold'])]
data = removeNanRows(data)

data['moldRow'] = [int(i) for i in data['moldRow']] # convert to int

print('added moldRow')
print(data)

print('testing moldRow')
print('does {} == {}?'.format(
    data.loc[1000, 'mold'], # expected mold of scan 1000
    data.loc[data.loc[1000, 'moldRow'], 'mold'])) # the mold located at expected mold's row of scan 1000

# plt.plot(data.loc[:, 'moldRow'])
# plt.show()

# -----------------------------------------------------------------------

def getNormalFromIdx(str, i):
    n = data.loc[i, [str+'A', str+'B', str+'C']].values
    n = n / np.sqrt(np.sum(n*n))
    return n

def computeNormalsDot(str, i, j):
    ni = getNormalFromIdx(str, i)
    nj = getNormalFromIdx(str, j)
    d = np.dot(ni, nj)
    return d

data['backDot'] = [computeNormalsDot('back', i, data.loc[i, 'moldRow']) for i in data.index]

print('computed normals dot')
print(data)

# -----------------------------------------------------------------------

# def computeControlNormalsDot(str, i):
#     N = data.shape[0]
#     d = 0
#     nbRandomTrials = 10
#     for i in np.arange(nbRandomTrials):
#         ri = np.random.randint(N)
#         d += computeNormalsDot(str, i, data.index[ri])
    
#     return d / nbRandomTrials

# data['backControlDot'] = [computeControlNormalsDot('back', i) for i in data.index]

# -----------------------------------------------------------------------

plt.show()
