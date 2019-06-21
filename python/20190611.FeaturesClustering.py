import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as DA # 'pip install -U scikit-learn', or 'conda install scikit-learn'
from sklearn.decomposition import PCA

import scipy.spatial
import scipy.cluster

import pandas as pd

matplotlib.style.use(['dark_background'])
plt.rcParams['grid.color'] = '#333333'
plt.rcParams['figure.facecolor'] = '#222222'
plt.rcParams['axes.facecolor'] = '#222222'
plt.rcParams["font.size"] = "10"
plt.rcParams["font.family"] = "consolas"

plt.close('all')

# -----------------------------------------------------------------------
# Tools for plotting in 3D

from mpl_toolkits.mplot3d import Axes3D

def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    '''
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)

# -----------------------------------------------------------------------

INCLUDE_SCANS = False
INCLUDE_MOLD_SCANS = False
ADD_RANDOM_FEATURE = False

# -----------------------------------------------------------------------
# Some feature subsets. Select one to use.

featureSubset_OriginalSpecs = np.array([ 'eside', 'eback', 'efront', 'ematerial', 'ebathtype', 'elength'])
featureSubset_OriginalMeasures = np.array([ 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8'])
featureSubset_K = np.array([ 'backK', 'frontK', 'bottomK', 'topK', 'rightK', 'leftK'])
featureSubset_D = np.array([ 'backD', 'frontD', 'bottomD', 'topD', 'rightD', 'leftD'])
featureSubset_DK = np.array([ 'backD', 'frontD', 'bottomD', 'topD', 'rightD', 'leftD', 'backK', 'frontK', 'bottomK', 'topK', 'rightK', 'leftK'])
featureSubset_MainNormalComponents = np.array([ 'backB', 'frontB', 'bottomA', 'topA', 'rightA', 'rightB', 'leftA', 'leftB'])
featureSubset_MainPlaneComponents = np.array([ 'backB', 'backD', 'frontB', 'frontD', 'bottomA', 'bottomD', 'topA', 'topD', 'rightA', 'rightB', 'rightD', 'leftA', 'leftB', 'leftD'])
featureSubset_ABC = np.array([ 'backA', 'backB', 'backC', 'frontA', 'frontB', 'frontC', 'bottomA', 'bottomB', 'bottomC', 'topA', 'topB', 'topC', 'rightA', 'rightB', 'rightC', 'leftA', 'leftB', 'leftC'])
featureSubset_ABCD = np.array([ 'backA', 'backB', 'backC', 'backD', 'frontA', 'frontB', 'frontC', 'frontD', 'bottomA', 'bottomB', 'bottomC', 'bottomD', 'topA', 'topB', 'topC', 'topD', 'rightA', 'rightB', 'rightC', 'rightD', 'leftA', 'leftB', 'leftC', 'leftD'])
featureSubset_ABCDK = np.array([ 'backK', 'frontK', 'bottomK', 'topK', 'rightK', 'leftK', 'backA', 'backB', 'backC', 'backD', 'frontA', 'frontB', 'frontC', 'frontD', 'bottomA', 'bottomB', 'bottomC', 'bottomD', 'topA', 'topB', 'topC', 'topD', 'rightA', 'rightB', 'rightC', 'rightD', 'leftA', 'leftB', 'leftC', 'leftD'])
featureSubset_LdaTrialNope = np.array([ 'topB', 'bottomB', 'bottomK', 'topK', 'rightK', 'leftK'])
featureSubset_UncorrelateABCDK = np.array([ 'backK', 'frontK', 'bottomK', 'topK', 'leftK', 'backA', 'backC', 'frontA', 'frontC', 'frontD', 'bottomB', 'bottomC', 'bottomD', 'topB', 'topD', 'rightA', 'rightB', 'rightD', 'leftA'])
featureSubset_ABCD_NoTopB = np.array([ 'backA', 'backB', 'backC', 'backD', 'frontA', 'frontB', 'frontC', 'frontD', 'bottomA', 'bottomB', 'bottomC', 'bottomD', 'topA', 'topC', 'topD', 'rightA', 'rightB', 'rightC', 'rightD', 'leftA', 'leftB', 'leftC', 'leftD'])

# Selection
featureSubset = featureSubset_ABCD

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

planeNames = ['back', 'front', 'top', 'bottom', 'left', 'right']

# -----------------------------------------------------------------------

# # Remove outliers.
# outlierIds = [0, 281, 311, 321, 362, 419, 476, 544, 557, 585, 588, 599, 624, 627, 1680, 3071, 3332, 3338, 7514, 7586, 8243, 8303, 8393 ]
# mustRemoveId = []
# for id in list(inputData[:,0]):
#     mustRemoveId.append(int(id) not in outlierIds)

# inputData = inputData[mustRemoveId]
# nbSamples = inputData.shape[0]

# -----------------------------------------------------------------------




# -----------------------------------------------------------------------
# Added a random feature, to see if we can detect that it brings nothing 
# to the data representation.

# if ADD_RANDOM_FEATURE:
#     randFeature = 20 + 30*np.random.rand(nbSamples, 1) # fits 8 measures range
#     # randFeature = np.random.rand(nbSamples, 1) # fits normals features range

#     inputData = np.hstack((inputData, randFeature))
#     featureNames = np.hstack((featureNames, 'rand'))

#     # Update data size.
#     nbFeatures = len(featureNames)

#%%

# Use a pandas dataframe to have access to useful data visualizations.
df = data[featureSubset]
df.hist(bins=50)

if (len(featureSubset) < 20): 
    pd.plotting.scatter_matrix(df, hist_kwds={'bins': 30})

#%%

def computeClustering(data, labels, figsize, distMetric, linkMethod, dendrogramOrientation):
    showDendro = False
    if showDendro: plt.figure(figsize=figsize) # for the dendrogram

    # Agglomerative hierarichal clustering.
    dist = scipy.spatial.distance.pdist(data, distMetric)
    links = scipy.cluster.hierarchy.linkage(dist, linkMethod)
    dendro = scipy.cluster.hierarchy.dendrogram(links, labels=labels, orientation=dendrogramOrientation, no_plot=not showDendro)
    
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # Convert to square matrix.
    dist = scipy.spatial.distance.squareform(dist)

    # Array of indices for sorting in same order as dendrogram leaves (close data grouped together).
    sortidx = np.array(dendro['leaves'])

    return (dist, sortidx)

def showDistMatrix(dist, labels, sortidx, thresh = None):
    dist = dist[np.ix_(sortidx, sortidx)]
    labels = labels[sortidx]

    range = np.arange(dist.shape[0])

    fig, ax = plt.subplots(figsize=(13,10))
    im = ax.matshow(dist)
    ax.set_yticks(range)
    ax.set_yticklabels(labels, fontsize=6)
    fig.colorbar(im)
    fig.tight_layout()

    if thresh != None:
        im.set_cmap('viridis_r')
        for it in ax.get_images():
            it.set_clim(0, thresh)

def showScanToMoldMap(map, sortidx):
    plt.plot(np.argsort(sortidx)[map[sortidx]], np.arange(len(sortidx)), '.-', alpha=0.8, linewidth=0.5, markersize=2)

def showDataMatrix(data, xLabels, yLabels, xSort, ySort):
    data = data[np.ix_(ySort, xSort)]
    xLabels = xLabels[xSort]
    yLabels = yLabels[ySort]

    dataforvis = data
    dataforvis = (dataforvis - dataforvis.mean(axis=0)) / dataforvis.std(axis=0)

    fig, ax = plt.subplots(figsize=(4,10))
    im = ax.matshow(dataforvis)
    ax.set_xticks(np.arange(len(xLabels)))
    ax.set_yticks(np.arange(len(yLabels)))
    ax.set_xticklabels(xLabels, rotation=90, fontsize=6)
    ax.set_yticklabels(yLabels, fontsize=6)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_aspect('auto')
    fig.colorbar(im)
    fig.tight_layout()

def computeMatchingDistances(dist):
    x = []
    y = []
    r = []
    for i in np.arange(nbSamples):
        if i != mapScanToMoldIdx[i]: # skip molds, since they match themselves
            x.append(sampleNames[i])
            y.append(dist[i, mapScanToMoldIdx[i]])
            r.append(np.searchsorted(np.sort(dist[i, :]), y[-1]) )

    N = len(x)

    if N == 0: return

    idx = np.argsort(r)
    x = np.array(x)[idx]
    y = np.array(y)[idx]
    r = np.array(r)[idx]
    
    fig = plt.figure(figsize=(12,6))

    ax = plt.subplot(2,1,1)
    ax.bar(np.arange(len(x)), y)
    ax.set_ylabel('expected mold distance')
    ax.set_xlim([0,len(x)])

    ax = plt.subplot(2,1,2)
    ax.bar(np.arange(len(x)), r)
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x, rotation=90, fontsize=6)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlim([0,len(x)])
    ax.set_ylim([0,50])
    ax.set_ylabel('expected mold rank')

    fig.tight_layout()

def computeLda(data, classLabels):
    lda = DA()
    lda.fit(data, classLabels)

    nbFeatures = lda.scalings_.shape[0]
    nbComponents = lda.scalings_.shape[1]

    weightMatrix = np.dot(np.ones((nbFeatures, 1)), np.array([lda.explained_variance_ratio_]))
    weightedScalings = lda.scalings_ * weightMatrix

    ldaData = lda.transform(data)

    fig = plt.figure()

    ax = plt.subplot(2,2,1)
    plt.title('LDA coef')
    im = ax.matshow(lda.coef_)
    ax.set_xticks(np.arange(nbFeatures))
    ax.set_yticks(np.arange(nbCategories))
    ax.set_xticklabels(featureNames, rotation=90, fontsize=8)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('features')
    ax.set_ylabel('classes')
    ax.set_aspect('auto')
    fig.colorbar(im)

    ax = plt.subplot(2,2,2)
    plt.title('LDA (weighted) scalings w: X* = X.w')
    im = ax.matshow(weightedScalings)
    ax.set_xticks(np.arange(nbComponents))
    ax.set_yticks(np.arange(nbFeatures))
    # ax.set_yticklabels(featureNames, fontsize=8)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('components')
    ax.set_ylabel('features')
    ax.set_aspect('auto')
    fig.colorbar(im)

    # ax = plt.subplot(2,2,3, projection='3d')
    # plt.title('LDA space representation 3D')
    # im = ax.scatter(ldaData[:,0], ldaData[:,1], ldaData[:,2], c=classLabels, alpha=0.5, cmap='nipy_spectral')
    # set_axes_equal(ax)
    # ax.set_xlabel('component 1')
    # ax.set_ylabel('component 2')
    # ax.set_zlabel('component 3')
    # fig.colorbar(im)

    ax = plt.subplot(2,2,3)
    plt.title('LDA space representation 2D')
    im = ax.scatter(ldaData[:,0], ldaData[:,1], c=classLabels, alpha=0.5, cmap='nipy_spectral')
    ax.set_xlabel('component 1')
    ax.set_ylabel('component 2')
    fig.colorbar(im)

    ax = plt.subplot(2,2,4)
    plt.title('explained variance')
    ax.bar(np.arange(nbComponents), lda.explained_variance_ratio_)
    plt.xlabel('components')

    fig.tight_layout()

    return ldaData[:,:nbComponents]

def computePca(data, classLabels):
    nbComponents = 8

    # From the docs: "The input data is centered but not scaled for each feature before applying the SVD."
    pca = PCA(n_components=nbComponents)

    pca.fit(data)
    pcaData = pca.transform(data)

    fig = plt.figure()

    # ax = plt.subplot(2,2,1,projection='3d')
    # plt.title('PCA space representation 3D')
    # im = ax.scatter(pcaData[:,0], pcaData[:,1], pcaData[:,2], c=classLabels, alpha=0.5, cmap='nipy_spectral')
    # ax.set_xlabel('component 1')
    # ax.set_ylabel('component 2')
    # ax.set_zlabel('component 3')
    # set_axes_equal(ax)
    # fig.colorbar(im)

    ax = plt.subplot(2,2,1)
    plt.title('PCA space representation 2D')
    im = ax.scatter(pcaData[:,0], pcaData[:,1], c=classLabels, alpha=0.5, cmap='nipy_spectral')
    ax.set_xlabel('component 1')
    ax.set_ylabel('component 2')
    fig.colorbar(im)

    ax = plt.subplot(2,2,2)
    plt.title('PCA components w^T: X* = X.w^T')
    im = ax.matshow(pca.components_.T)
    ax.set_xticks(np.arange(nbComponents))
    ax.set_yticks(np.arange(nbFeatures))
    ax.set_yticklabels(featureNames, fontsize=8)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('components')
    ax.set_ylabel('features')
    ax.set_aspect('auto')
    fig.colorbar(im)

    ax = plt.subplot(2,2,4)
    plt.title('explained variance')
    ax.bar(np.arange(nbComponents), pca.explained_variance_ratio_)
    plt.xlabel('components')

    fig.tight_layout()

    return pcaData[:,:nbComponents]

# -----------------------------------------------------------------------

distSamples, sortSamples = computeClustering(inputData, sampleNames, (2,10), 'euclidean', 'ward', 'right')
distFeatures, sortFeatures = computeClustering(inputData.T, featureNames, (10,2), 'correlation', 'complete', 'top')

sortCategories = np.argsort(categories)

computeMatchingDistances(distSamples)

# Finds a non-supervised representation of data.
pcaData = computePca(inputData, categories)

# Finds a supervised (uses class labels) representation of data that maximizes clusters discrimination.
ldaData = computeLda(inputData, categories)
# ldaData = computeLda(pcaData[:,:3], categories)########
distLda, sortLda = computeClustering(ldaData, sampleNames, (2,10), 'euclidean', 'ward', 'right')

computeMatchingDistances(distLda)

# -----------------------------------------------------------------------

showDistMatrix(distSamples, sampleNames, sortSamples)
showScanToMoldMap(mapScanToMoldIdx, sortSamples)
plt.xlabel('samples distance matrix')

# showDistMatrix(distFeatures, featureNames, sortFeatures)
showDistMatrix(distFeatures, featureNames, np.arange(nbFeatures))
plt.xlabel('features distance matrix')

showDistMatrix(distLda, sampleNames, sortLda)
showScanToMoldMap(mapScanToMoldIdx, sortLda)
plt.xlabel('samples distance matrix in LDA space')

plt.show()

theend = 0