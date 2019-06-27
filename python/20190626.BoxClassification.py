#%%
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

INCLUDE_SCANS = True
INCLUDE_MOLD_SCANS = False

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
featureSubset_box2 = ['heightFront', 'heightBack', 'lengthTop', 'lengthDown', 'slopeBack', 'parallelismTop', 'parallelismDown', 'frontK', 'front-planarity', 'backK', 'back-planarity', 'leftK', 'left-planarity', 'rightK', 'right-planarity']

# Selection
featureSubset = featureSubset_box2

#%%
# Create the main data matrix.

# Merge scans and molds
moldData = pd.read_csv('python/data/20190617.planes.molds.csv', index_col=False)
moldData['type'] = 'mold'
data = moldData

if INCLUDE_SCANS:
    scanData = pd.read_csv("python/data/20190617.planes.goodscans.csv", index_col=False)
    scanData['type'] = 'scan'
    data = data.append(scanData, ignore_index=True)

if INCLUDE_MOLD_SCANS:
    moldScanData = pd.read_csv('python/data/20190617.planes.moldscans.csv', index_col=False)
    moldScanData['type'] = 'moldscan'
    data = data.append(moldScanData, ignore_index=True)

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
    N = data.values.shape[0]

    # Add specs category column
    specs = data.values[:, 2:8]
    categorieNames = ["%i%i%i%i%i%i" %
        (specs[i, 0], specs[i, 1], specs[i, 2], specs[i, 3], specs[i, 4], specs[i, 5]) for i in np.arange(N)]
    categorieIds = [int(c) for c in categorieNames]
    categorieUniqueIds = [x for i, x in enumerate(categorieIds) if i == categorieIds.index(x)]
    categories = [categorieUniqueIds.index(c) for c in categorieIds]

    data['specs-category'] = categories

    # Extract samples id from first column.
    ids = data.values[:, 0:2]
    data['name'] = ["%6i %3i %4i-%3i" % (categorieIds[i], categories[i], ids[i, 0], ids[i, 1]) for i in np.arange(N)]

    return data

data = addSpecsCategory(data)

#%%

# -----------------------------------------------------------------------

# Remove outliers.
outlierIds = [0, 281, 311, 321, 362, 419, 476, 544, 557, 585, 588, 599, 624, 625, 627, 676, 1680, 1843, 1853, 1974, 1979, 2508, 2703, 3071, 3332, 3338, 7514, 7586, 8243, 8303, 8393 ]
toKeep = [i not in outlierIds for i in data['id']]
print('Removing:')
print(data[[not i for i in toKeep]])
data = data[toKeep]

#%%

print(data.head())

def pandasPlot(data):
    data.hist(bins=50)

    if (len(featureSubset) < 20): 
        pd.plotting.scatter_matrix(data, hist_kwds={'bins': 30})

pandasPlot(data[featureSubset])

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
testi = min(700, data.values.shape[0]-1)
print('does {} == {}?'.format(
    data.loc[testi, 'mold'], # expected mold of scan 1000
    data.loc[data.loc[testi, 'moldRow'], 'mold'])) # the mold located at expected mold's row of scan 1000

#%%

def computeDiffWithExpectedMold(feat):
    return np.array([data.loc[i, feat] - data.loc[data.loc[i, 'moldRow'], feat] for i in data.index])

# Determine features 'quality': if the variance of the difference between the 
# scan and the expected mold is small compared to the variance of the feature.
featDiffNames = []
featureClassificationQuality = []
for feat in featureSubset:
    # Feature name for the difference.
    featDiff = feat + '-diff'
    featDiffNames.append(featDiff)
    # Compute difference.
    data[featDiff] = computeDiffWithExpectedMold(feat)
    # Compute the quality.
    scanData = data[data['type'] == 'scan']
    featureClassificationQuality.append(scanData[feat].std() / scanData[featDiff].std())

# Show differences histograms.
data.loc[data['type'] == 'scan', featDiffNames].hist(bins=50)

# Plot the quality for each feature.
plt.figure()
plt.bar(np.arange(len(featureSubset)), featureClassificationQuality)
plt.xticks(np.arange(len(featureSubset)), featureSubset, rotation=80)
plt.ylabel('classification potential')

#%%

# Difference thresholds determined from the histograms of the differences
# with the expected molds. Thresholding difference with this value should
# keep near 100% of the expected mold.
classThresholds = {
    'heightFront': 0.03,
    'lengthDown': 0.1,
    'slopeBack': 0.15,
    'parallelismTop': 0.06,
    # 'leftK': 0.0035,
}

# Normalize the feature with the difference threshold. The difference can 
# than be compared to 1 for thresholding.
for feat in classThresholds.keys():
    data[feat + '-norm'] = data[feat] / classThresholds[feat]

#%%

def computeScanRank(scanIdx, moldData, features):
    # Compute euclidean distances.
    distWithAllMolds = np.zeros(moldData.shape[0])
    for feat in features:
        diff = moldData.loc[:, feat + '-norm'] - scanData.loc[scanIdx, feat + '-norm']
        diff = diff.values
        distWithAllMolds = distWithAllMolds + diff*diff
    distWithAllMolds = np.sqrt(distWithAllMolds)

    # Compute rank.
    sortIdx = np.argsort(distWithAllMolds)
    rank = sortIdx[sortIdx]
    moldIdx = scanData.loc[scanIdx, 'moldRow']
    if moldIdx in moldData.index:
        imold = moldData.index.get_loc(moldIdx)
        return rank[imold]
    else: # not in list
        return 600

#%%
# Analyze the effect of each classification feature.

scanData = data.loc[data['type'] == 'scan']
moldData = data.loc[data['type'] == 'mold']

nbMolds = moldData.shape[0]
nbScans = scanData.shape[0]

classSize = np.zeros((nbScans, len(classThresholds) + 1))
expectedMoldIsInList = np.ones((nbScans, len(classThresholds) + 1))
ranks = np.zeros((nbScans, len(classThresholds) + 1))

for i, idx in enumerate(scanData.index):
    mask = np.ones(nbMolds) == 1 # mask filter to filter molds
    classSize[i, 0] = np.sum(mask)
    ranks[i, 0] = computeScanRank(idx, moldData.loc[mask, :], classThresholds.keys())
    for j, feat in enumerate(classThresholds.keys()):
        # Find molds whose 
        diff = np.abs(moldData[feat + '-norm'].values - scanData.loc[idx, feat + '-norm'])
        thresh = diff < 1.0 # has been normalized
        mask = np.logical_and(mask, thresh)

        classSize[i, j+1] = np.sum(mask)
        expectedMoldIsInList[i, j+1] = mask[moldData.index.get_loc(scanData.loc[idx, 'moldRow'])]
        ranks[i,j+1] = computeScanRank(idx, moldData.loc[mask, :], classThresholds.keys())

#%%

plt.figure(figsize=(10,6))
x = np.arange(nbScans)
si = np.argsort(classSize[:, -1])
for i in np.arange(classSize.shape[1]):
    plt.bar(x, classSize[si, i])
    performance = np.sum(expectedMoldIsInList[:, i]) / nbScans
    print('Performance at classification {}: {}%'.format(i, performance * 100))

plt.xticks(x, scanData.iloc[si,:]['name'], rotation=80)

#%%

plt.figure(figsize=(10,6))
# ri = np.argsort(ranks[:, -1])
for i in np.arange(ranks.shape[1]):
    ri = np.argsort(ranks[:, i])
    plt.bar(x, ranks[ri, i], alpha=0.8)
# plt.xticks(x, scanData.iloc[ri,:]['name'], rotation=80) # not if sorting each time
plt.ylim([0,100])
plt.xlabel('sorted scan')
plt.ylabel('expected mold rank')

# -----------------------------------------------------------------------

plt.show()

theend = 0