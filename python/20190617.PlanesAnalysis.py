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

#%%

scanData = pd.read_csv("python/data/20190617.planes.goodscans.csv", index_col=False)
moldData = pd.read_csv('python/data/20190617.planes.molds.csv', index_col=False)
moldScanData = pd.read_csv('python/data/20190617.planes.moldscans.csv', index_col=False)

scanData['type'] = 'scan'
moldData['type'] = 'mold'
moldScanData['type'] = 'moldscan'

print('added data type column')

data = scanData
data = data.append(moldData, ignore_index=True)
# data = data.append(moldScanData, ignore_index=True)

print('appended into a single dataframe')

#%%

def removeNanRows(df):
    nanRows = df.isna().any(axis=1)

    print('removing following NaN rows:')
    print(df[nanRows == True])

    return df[nanRows != True]

data = removeNanRows(data)

# Remove id 0, which is the scan when generating the molds data.
data = data[data['id'] > 0]

#%%

def addSpecsCategory(data):
    # Add specs category column
    specs = data.values[:, 2:8]
    categorieNames = ["%i%i%i%i%i%i" %
        (specs[i, 0], specs[i, 1], specs[i, 2], specs[i, 3], specs[i, 4], specs[i, 5]) for i in np.arange(data.values.shape[0])]
    categorieIds = [int(c) for c in categorieNames]
    categorieUniqueIds = [x for i, x in enumerate(categorieIds) if i == categorieIds.index(x)]
    categories = [categorieUniqueIds.index(c) for c in categorieIds]
    data['specs-category'] = categories
    return data

data = addSpecsCategory(data)

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
    data.loc[700, 'mold'], # expected mold of scan 1000
    data.loc[data.loc[700, 'moldRow'], 'mold'])) # the mold located at expected mold's row of scan 1000

#%%

planeNames = ['back', 'front', 'top', 'bottom', 'left', 'right']

planeAngleDiffNames = [plane + '-angle-diff' for plane in planeNames]
planeDistDiffNames = [plane + '-dist-diff' for plane in planeNames]
planeCentroiDiffNames = [plane + '-centroid-diff' for plane in planeNames]
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

def getCentroidFromIdx(plane, i):
    c = data.loc[i, [plane+'-cx', plane+'-cy', plane+'-cz']].values * 100 # convert to cm
    return c

def computeCentroidDiff(plane, i, j):
    ci = getCentroidFromIdx(plane, i)
    cj = getCentroidFromIdx(plane, j)
    d = ci-cj
    d = np.sqrt(np.sum(d*d))
    return d

for idx, plane in enumerate(planeNames):
    data[planeCentroiDiffNames[idx]] = [computeCentroidDiff(plane, i, data.loc[i, 'moldRow']) for i in data.index]

print('computed centroid diff')
print(data[planeCentroiDiffNames].tail())

#%%

# Plot hists by plane 

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
distBins = np.arange(0, 20, 0.5)

xlabelAngle = 'angles (degree)'
xlabelDist = 'distance (cm)'

plot(planeAngleDiffNames, angleBins, xlabelAngle)
plot(planeCentroiDiffNames, distBins, xlabelDist)

#%%

# Plot global hists 

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.hist(getScanDataValues(planeAngleDiffNames).flatten(), bins=angleBins, alpha=alpha)
plt.hist(getMoldScanDataValues(planeAngleDiffNames).flatten(), bins=angleBins, alpha=alpha)
plt.xlabel(xlabelAngle)

plt.subplot(1,2,2)
plt.hist(getScanDataValues(planeCentroiDiffNames).flatten(), bins=distBins, alpha=alpha)
plt.hist(getMoldScanDataValues(planeCentroiDiffNames).flatten(), bins=distBins, alpha=alpha)
plt.xlabel(xlabelDist)

plt.legend(['scan', 'mold'])
plt.tight_layout()

#%%

# Plot D plane parameter

# # plt.figure()
# # plt.subplot(2,1,1)
# data.loc[data['type'] == 'moldscan', ['backD', 'frontD', 'topD', 'bottomD', 'rightD', 'leftD']].plot()
# plt.ylim([-0.8, 0.8])

# # plt.subplot(2,1,2)
# data.loc[data['type'] == 'scan', ['backD', 'frontD', 'topD', 'bottomD', 'rightD', 'leftD']].plot()
# plt.ylim([-0.8, 0.8])

# data.loc[data['type'] == 'moldscan', ['back-dist-diff', 'front-dist-diff', 'top-dist-diff', 'bottom-dist-diff', 'right-dist-diff', 'left-dist-diff']].plot()
# plt.ylim([-80, 80])


#%%

# Trying to groupby specs categories

# subdata = data.loc[data['type'] == 'scan', ['top-dist-diff', 'specs-category']]
# for d in subdata.groupby('specs-category'):
#     print(d.tail

# plt.plot(data.loc[data['type'] == 'scan', ['top-dist-diff', 'specs-category']].groupby('specs-category').mean())

# data.loc[data['type'] == 'scan', ['top-dist-diff' + 'specs-category'].groupby('specs-category').hist('top-dist-diff')


#%%

plt.show()
