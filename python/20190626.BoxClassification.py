#%%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# For this to work in interactive Python, set "Notebook File Root" setting to ${fileDirname}
from initdata import *

# -----------------------------------------------------------------------
# Some feature subsets. Select one to use.

featureSubset_OriginalSpecs = [ 'eside', 'eback', 'efront', 'ematerial', 'ebathtype', 'elength']
featureSubset_OriginalMeasures = [ 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8']
featureSubset_K = [ 'backK', 'frontK', 'bottomK', 'topK', 'rightK', 'leftK']
featureSubset_D = [ 'backD', 'frontD', 'bottomD', 'topD', 'rightD', 'leftD']
featureSubset_DK = [ 'backD', 'frontD', 'bottomD', 'topD', 'rightD', 'leftD', 'backK', 'frontK', 'bottomK', 'topK', 'rightK', 'leftK']
featureSubset_MainNormalComponents = [ 'backB', 'frontB', 'bottomA', 'topA', 'rightA', 'rightB', 'leftA', 'leftB']
featureSubset_MainPlaneComponents = [ 'backB', 'backD', 'frontB', 'frontD', 'bottomA', 'bottomD', 'topA', 'topD', 'rightA', 'rightB', 'rightD', 'leftA', 'leftB', 'leftD']
featureSubset_ABC = [ 'backA', 'backB', 'backC', 'frontA', 'frontB', 'frontC', 'bottomA', 'bottomB', 'bottomC', 'topA', 'topB', 'topC', 'rightA', 'rightB', 'rightC', 'leftA', 'leftB', 'leftC']
featureSubset_ABCD = [ 'backA', 'backB', 'backC', 'backD', 'frontA', 'frontB', 'frontC', 'frontD', 'bottomA', 'bottomB', 'bottomC', 'bottomD', 'topA', 'topB', 'topC', 'topD', 'rightA', 'rightB', 'rightC', 'rightD', 'leftA', 'leftB', 'leftC', 'leftD']
featureSubset_ABCDK = [ 'backK', 'frontK', 'bottomK', 'topK', 'rightK', 'leftK', 'backA', 'backB', 'backC', 'backD', 'frontA', 'frontB', 'frontC', 'frontD', 'bottomA', 'bottomB', 'bottomC', 'bottomD', 'topA', 'topB', 'topC', 'topD', 'rightA', 'rightB', 'rightC', 'rightD', 'leftA', 'leftB', 'leftC', 'leftD']
featureSubset_LdaTrialNope = [ 'topB', 'bottomB', 'bottomK', 'topK', 'rightK', 'leftK']
featureSubset_UncorrelateABCDK = [ 'backK', 'frontK', 'bottomK', 'topK', 'leftK', 'backA', 'backC', 'frontA', 'frontC', 'frontD', 'bottomB', 'bottomC', 'bottomD', 'topB', 'topD', 'rightA', 'rightB', 'rightD', 'leftA']
featureSubset_ABCD_NoTopBottomABC = [ 'backA', 'backB', 'backC', 'backD', 'frontA', 'frontB', 'frontC', 'frontD', 'bottomD', 'topD', 'rightA', 'rightB', 'rightC', 'rightD', 'leftA', 'leftB', 'leftC', 'leftD']
featureSubset_ABCD_NoTopBottomABC = [ 'backA', 'backB', 'backC', 'backD', 'frontA', 'frontB', 'frontC', 'frontD', 'bottomD', 'topD', 'rightA', 'rightB', 'rightC', 'rightD', 'leftA', 'leftB', 'leftC', 'leftD']

featureSubset_box1 = ['heightFront', 'heightBack', 'heightRatio', 'lengthTop', 'lengthDown', 'lengthRatio', 'slopeBack', 'slopeFront', 'slopeDown', 'slopeSide', 'parallelismTop', 'parallelismDown', 'parallelismRatio']
featureSubset_box2 = ['heightFront', 'heightBack', 'lengthTop', 'lengthDown', 'slopeBack', 'widthTopFront', 'widthTopBack', 'widthDownFront', 'widthDownBack', 'widthTopRatio', 'widthDownRatio', 'parallelismTop', 'parallelismDown', 'leftK', 'rightK']
featureSubset_box3 = ['heightFront', 'heightBack', 'lengthTop', 'lengthDown', 'slopeBack', 'parallelismTop', 'parallelismDown', 'frontK', 'front-planarity', 'backK', 'back-planarity', 'leftK', 'left-planarity', 'rightK', 'right-planarity']

# Selection
featureSubset = featureSubset_box2

#%%

data = importAndPreprocessData(featureSubset, includeScans=True, includeMoldScans=False, outlierScansStd=3, subsampleScans=3)

def getMoldData(): return data.loc[data['type'] == 'mold', :]
def getScanData(): return data.loc[data['type'] == 'scan', :]

#%%
print("Computing features discriminant quality...")

def computeDiffWithExpectedMold(feat):
    return np.array([data.loc[i, feat] - data.loc[data.loc[i, 'moldRow'], feat] for i in data.index])

# Determine features 'quality': if the variance of the difference between the 
# scan and the expected mold is small compared to the variance of the feature.
featDiffNames = []
featureClassificationQuality = []
for feat in featureSubset:
    # Feature name for the difference.
    featDiff = feat + '-diff-norm'
    featDiffNames.append(featDiff)
    # Compute difference.
    moldStd = getMoldData()[feat].std()
    data[featDiff] = computeDiffWithExpectedMold(feat) / moldStd
    # Compute the quality.
    scanDiffStd = getScanData()[featDiff].std()
    featureClassificationQuality.append(1 / scanDiffStd)

print("Plotting features discriminant quality...")

# Show differences histograms.
getScanData()[featDiffNames].hist(bins=np.arange(-1,1,0.05), figsize=(10,10))

# Plot the quality for each feature.
plt.figure()
plt.bar(np.arange(len(featureSubset)), featureClassificationQuality)
plt.xticks(np.arange(len(featureSubset)), featureSubset, rotation=80)
plt.ylabel('classification potential')

#%%
print("Normalizing data with defined thresholds...")

# Difference thresholds determined from the histograms of the differences
# with the expected molds. Thresholding difference with this value should
# keep near 100% of the expected mold.
classThresholds = {
    'heightFront': 0.03,
    'lengthDown': 0.1,
    'slopeBack': 0.15,
    'parallelismTop': 0.06,
    'widthDownBack': 0.05,
    # 'leftK': 0.0035,
}

# Normalize the feature with the difference threshold. The difference can 
# then be compared to 1 for thresholding.
featureNormNames = []
for feat in classThresholds.keys():
    featureNormNames.append(feat + '-norm')
    data[featureNormNames[-1]] = data[feat] / classThresholds[feat]

#%%
print("Computing ranks...")

def computeScanRank(scanIdx, moldData, features):
    # Compute euclidean distances.
    distWithAllMolds = np.zeros(moldData.shape[0])
    for feat in features:
        diff = moldData.loc[:, feat] - getScanData().loc[scanIdx, feat]
        diff = diff.values
        distWithAllMolds = distWithAllMolds + diff*diff
    distWithAllMolds = np.sqrt(distWithAllMolds)

    # Compute rank.
    sortIdx = np.argsort(distWithAllMolds)
    rank = sortIdx[sortIdx]
    moldIdx = getScanData().loc[scanIdx, 'moldRow']
    if moldIdx in moldData.index:
        imold = moldData.index.get_loc(moldIdx)
        return rank[imold]
    else: # not in list
        return 600

#%%
print("Computing ranks for each feature...")

# Analyze the effect of each classification feature.

nbMolds = getMoldData().shape[0]
nbScans = getScanData().shape[0]

classSize = np.zeros((nbScans, len(featureNormNames) + 1))
expectedMoldIsInList = np.ones((nbScans, len(featureNormNames) + 1))
ranks = np.zeros((nbScans, len(featureNormNames) + 2))

for i, idx in enumerate(getScanData().index):
    mask = np.ones(nbMolds) == 1 # mask filter to filter molds
    classSize[i, 0] = np.sum(mask)
    ranks[i, 0] = computeScanRank(idx, getMoldData().loc[mask, :], featureNormNames)
    for j, feat in enumerate(featureNormNames):
        # Find molds whose 
        diff = np.abs(getMoldData()[feat].values - getScanData().loc[idx, feat])
        thresh = diff < 1.0 # has been normalized
        mask = np.logical_and(mask, thresh)

        classSize[i, j+1] = np.sum(mask)
        expectedMoldIsInList[i, j+1] = mask[getMoldData().index.get_loc(getScanData().loc[idx, 'moldRow'])]
        ranks[i,j+1] = computeScanRank(idx, getMoldData().loc[mask, :], featureNormNames)

#%%
print('Weighted euclidean distance performance...')

for i, feat in enumerate(featureSubset):
    data[feat + '-weighted'] = data[feat] * featureClassificationQuality[i]

featureWeightNames = [feat + '-weighted' for feat in classThresholds.keys()]

for i, idx in enumerate(getScanData().index):
    ranks[i, -1] = computeScanRank(idx, getMoldData(), featureWeightNames)

#%%
print("Plotting performance...")

def plotShortListSizeReferenceLine():
    plt.plot(20 * np.ones(nbScans))

plt.figure(figsize=(10,6))
x = np.arange(nbScans)
si = np.argsort(classSize[:, -1])
for i in np.arange(classSize.shape[1]):
    plt.plot(x, classSize[si, i], alpha=0.8)
    performance = np.sum(expectedMoldIsInList[:, i]) / nbScans
    print('Performance at classification {}: {}%'.format(i, performance * 100))

plt.ylabel('class size')

plotShortListSizeReferenceLine()

if (nbScans < 300): # avoid too much label
    plt.xticks(x, getScanData().iloc[si,:]['name'], rotation=80)

#%%
print("Plotting ranks...")

plt.figure(figsize=(10,6))
# ri = np.argsort(ranks[:, -1])
for i in np.arange(ranks.shape[1]):
    ri = np.argsort(ranks[:, i])
    plt.plot(100 * x / nbScans, ranks[ri, i], alpha=0.8)
# plt.xticks(x, getScanData().iloc[ri,:]['name'], rotation=80) # not if sorting each time
plt.xlabel('sorted scans (%)')
plt.ylabel('expected mold rank')
plt.xlim([0,100])
plt.ylim([0,100])

plotShortListSizeReferenceLine()

plt.legend(['unweighted euclidean'] + list(classThresholds.keys()) + ['weighted euclidean', 'short list size objective'])

# -----------------------------------------------------------------------

plt.show()

theend = 0