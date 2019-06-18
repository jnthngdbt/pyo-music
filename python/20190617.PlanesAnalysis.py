#%%

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd

matplotlib.style.use(['dark_background'])
plt.rcParams['grid.color'] = '#333333'
plt.rcParams['figure.facecolor'] = '#222222'
plt.rcParams['axes.facecolor'] = '#222222'
plt.rcParams["font.size"] = "11"
plt.rcParams["font.family"] = "consolas"

plt.close('all')

#%% Some feature subsets. Select one to use.

featureSubset_ABCD = [ 'id', 'mold', 'backA', 'backB', 'backC', 'backD', 'frontA', 'frontB', 'frontC', 'frontD', 'bottomA', 'bottomB', 'bottomC', 'bottomD', 'topA', 'topB', 'topC', 'topD', 'rightA', 'rightB', 'rightC', 'rightD', 'leftA', 'leftB', 'leftC', 'leftD']
featureSubset_ABCDK = [ 'id', 'mold', 'backK', 'frontK', 'bottomK', 'topK', 'rightK', 'leftK', 'backA', 'backB', 'backC', 'backD', 'frontA', 'frontB', 'frontC', 'frontD', 'bottomA', 'bottomB', 'bottomC', 'bottomD', 'topA', 'topB', 'topC', 'topD', 'rightA', 'rightB', 'rightC', 'rightD', 'leftA', 'leftB', 'leftC', 'leftD']

# Selection
featureSubset = featureSubset_ABCDK

#%%

scanDataFull = pd.read_csv("python/data/20190617.planes.scans.csv", index_col=False)
moldDataFull = pd.read_csv('python/data/20190617.planes.molds.csv', index_col=False)
moldScanDataFull = pd.read_csv('python/data/20190617.planes.moldscans.csv', index_col=False)

# print(scanDataFull)
# print(moldDataFull)
# print(moldScanDataFull)

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

#%%

scanData['type'] = 'scan'
moldData['type'] = 'mold'
moldScanData['type'] = 'moldscan'

print('added data type column')
# print(scanData)
# print(moldData)
# print(moldScanData)

data = scanData
data = data.append(moldData, ignore_index=True)
data = data.append(moldScanData, ignore_index=True)

print('appended into a single dataframe')
# print(data)

# data.groupby('type').mean().plot.bar()

# plt.show()

#%%

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
# print(data)

print('testing moldRow')
print('does {} == {}?'.format(
    data.loc[1000, 'mold'], # expected mold of scan 1000
    data.loc[data.loc[1000, 'moldRow'], 'mold'])) # the mold located at expected mold's row of scan 1000

# plt.plot(data.loc[:, 'moldRow'])
# plt.show()

#%%

planeNames = ['back', 'front', 'top', 'bottom', 'left', 'right']

planeAngleDiffNames = [plane + '-angle-diff' for plane in planeNames]
planeDistDiffNames = [plane + '-dist-diff' for plane in planeNames]
planeAngleBaselineNames = [plane + '-angle-baseline' for plane in planeNames]
planeDistBaselineNames = [plane + '-dist-baseline' for plane in planeNames]

def getNormalFromIdx(plane, i):
    n = data.loc[i, [plane+'A', plane+'B', plane+'C']].values
    n = n / np.sqrt(np.sum(n*n))
    return n

def computeNormalsDot(plane, i, j):
    ni = getNormalFromIdx(plane, i)
    nj = getNormalFromIdx(plane, j)
    d = min(1.0, np.dot(ni, nj))
    return d

def convertDotToAngleDeg(dot):
    a = 180.0 * np.arccos(dot) / np.pi
    return a

def computeNormalsAngleDeg(plane, i, j):
    return convertDotToAngleDeg(computeNormalsDot(plane, i, j))

for idx, plane in enumerate(planeNames):
    data[planeAngleDiffNames[idx]] = [computeNormalsAngleDeg(plane, i, data.loc[i, 'moldRow']) for i in data.index]

print('computed normals angles diff')
print(data[planeAngleDiffNames].tail())

#%%

def getDistFromIdx(plane, i):
    d = data.loc[i, plane+'D'] * 100 # convert to cm
    return d

def computeDistDiff(plane, i, j):
    di = getDistFromIdx(plane, i)
    dj = getDistFromIdx(plane, j)
    d = di-dj
    return d

for idx, plane in enumerate(planeNames):
    data[planeDistDiffNames[idx]] = [computeDistDiff(plane, i, data.loc[i, 'moldRow']) for i in data.index]

print('computed d (distance) diff')
print(data[planeDistDiffNames].tail())

#%%

# def computeAngleBaseline(plane, i):
#     N = data.shape[0]
#     a = 0
#     nbRandomTrials = 10
#     for i in np.arange(nbRandomTrials):
#         ri = np.random.randint(N)
#         a += convertDotToAngleDeg(computeNormalsDot(plane, int(i), data.index[ri]))
#     return a / nbRandomTrials

# for idx, plane in enumerate(planeNames):
#     data[planeAngleBaselineNames[idx]] = [computeAngleBaseline(plane, i) for i in data.index]

# print('computed angle baseline diff')
# print(data[planeAngleBaselineNames].tail())

#%%

# def computeDistBaseline(plane, i):
#     N = data.shape[0]
#     d = 0
#     nbRandomTrials = 10
#     for i in np.arange(nbRandomTrials):
#         ri = np.random.randint(N)
#         d += getDistFromIdx(plane, i) - getDistFromIdx(plane, ri)
#     return d / nbRandomTrials

# for idx, plane in enumerate(planeNames):
#     data[planeDistBaselineNames[idx]] = [computeDistBaseline(plane, i) for i in data.index]

# print('computed dist baseline diff')
# print(data[planeDistBaselineNames].tail())

#%%

def getMoldScanDataValues(colNames): return data.loc[data['type'] == 'moldscan', colNames].values
def getScanDataValues(colNames): return data.loc[data['type'] == 'scan', colNames].values

alpha = 0.5

def plot(colNames, bins, xlabel):
    moldScanDiff = getMoldScanDataValues(colNames)
    scanDiff = getScanDataValues(colNames)

    plt.figure(figsize=(8,8))
    for i, plane in enumerate(planeNames):
        plt.subplot(3,2,i+1)
        plt.hist(scanDiff[:,i], bins=bins, alpha=alpha)
        plt.hist(moldScanDiff[:,i], bins=bins, alpha=alpha)
        plt.title(plane)

        if i > 3:
            plt.xlabel(xlabel)

    plt.legend(['scan', 'mold'])
    plt.tight_layout()

angleBins = np.arange(0, 10, 0.2)
distBins = np.arange(-12, 12, 0.4)

xlabelAngle = 'angles (degree)'
xlabelDist = 'distance (cm)'

plot(planeAngleDiffNames, angleBins, xlabelAngle)
plot(planeDistDiffNames, distBins, xlabelDist)

#%%

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.hist(getScanDataValues(planeAngleDiffNames).flatten(), bins=angleBins, alpha=alpha)
plt.hist(getMoldScanDataValues(planeAngleDiffNames).flatten(), bins=angleBins, alpha=alpha)
plt.xlabel(xlabelAngle)

plt.subplot(1,2,2)
plt.hist(getScanDataValues(planeDistDiffNames).flatten(), bins=distBins, alpha=alpha)
plt.hist(getMoldScanDataValues(planeDistDiffNames).flatten(), bins=distBins, alpha=alpha)
plt.xlabel(xlabelDist)

plt.legend(['scan', 'mold'])
plt.tight_layout()


#%%

# Cases of the second mode of top/bottom D distribution
weird = data.loc[data['top-dist-diff'] > 5, ['id', 'mold']]
print(weird)
weird.hist()



plt.show()