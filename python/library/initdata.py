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

featureSubset_Size = ['heightFront', 'heightBack', 'heightRatio', 'lengthTop', 'lengthDown', 'lengthRatio', 'widthTopFront', 'widthTopBack', 'widthDownFront', 'widthDownBack', 'widthTopRatio', 'widthDownRatio']
featureSubset_SizeNoCorr = ['heightFront', 'lengthTop', 'widthDownFront', 'widthDownBack']
featureSubset_Angle = ['slopeBack', 'slopeFront', 'slopeDown', 'slopeSide', 'parallelismTop', 'parallelismDown', 'parallelismRatio']

def importAndPreprocessData(
    featureSubset, 
    includeScans = False, 
    includeMoldScans = False, 
    subsampleScans=3, 
    outlierScansStd=None, 
    showData=False,
    standardize=False):

    print("Creating main database...")

    # Merge scans and molds
    moldData = pd.read_csv('data/20190617.planes.molds.csv', index_col=False)
    moldData['type'] = 'mold'
    data = moldData

    if includeScans:
        scanData = pd.read_csv("data/20190617.planes.allscans.csv", index_col=False)
        scanData['type'] = 'scan'

        # Subsample
        scanData = scanData.iloc[np.arange(0, scanData.shape[0], subsampleScans), :]

        data = data.append(scanData, ignore_index=True)

    if includeMoldScans:
        moldScanData = pd.read_csv('data/20190617.planes.moldscans.csv', index_col=False)
        moldScanData['type'] = 'moldscan'
        data = data.append(moldScanData, ignore_index=True)

    def getMoldData(): return data.loc[data['type'] == 'mold', :]
    def getScanData(): return data.loc[data['type'] == 'scan', :]
    def getMoldScanData(): return data.loc[data['type'] == 'moldscan', :]
    def getFullScanData(): return data.loc[(data['type'] == 'moldscan') | (data['type'] == 'scan'), :]

    print('appended into a single dataframe')

    #%%
    print("Removing NaNs...")

    def removeNanRows(df):
        nanRows = df.isna().any(axis=1)

        print('removing following NaN rows:')
        print(df[nanRows == True])

        return df[nanRows != True]

    data = removeNanRows(data)

    # Remove id 0, which is the scan when generating the molds data.
    data = data[data['id'] > 0]

    #%%
    print("Adding original specs category...")

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
    if (outlierScansStd != None):
        print("Removing outliers...")

        from scipy import stats
        inliners = (np.abs(stats.zscore(data[featureSubset])) < outlierScansStd).all(axis=1)
        print(data[np.logical_not(inliners)])
        data = data[inliners]
    else:
        print("Skipping removing outliers...")

    #%%
    if (standardize):
        print("Standardize features...")

        for feat in featureSubset:
            data[feat] = (data[feat] - data[feat].mean(axis=0)) / data[feat].std(axis=0)

    #%%
    print("Showing mold priors...")

    data['mold'].hist(bins=np.arange(data['mold'].max()))

    #%%
    if showData:
        print("Showing data with pandas visualization...")

        axx = getFullScanData()[featureSubset].hist(bins=50, alpha=0.5, figsize=(10,10))
        axx = axx.flatten()
        axx = axx[:len(featureSubset)]
        axx = getMoldData()[featureSubset].hist(bins=50, alpha=0.5, ax=axx)
        plt.suptitle('scan then mold')

    #%%
        if (len(featureSubset) < 20): 
            # Subsample
            N = getMoldData().shape[0]
            d = getMoldData().iloc[np.arange(0, N, int(N / 100)), :] # subsample
            sm = pd.plotting.scatter_matrix(d[featureSubset], figsize=(10,10), hist_kwds={'bins': 30})
            #Change label rotation
            [s.xaxis.label.set_rotation(45) for s in sm.reshape(-1)]
            [s.yaxis.label.set_rotation(0) for s in sm.reshape(-1)]
            [s.get_yaxis().set_label_coords(-2.0,0.5) for s in sm.reshape(-1)]
            plt.suptitle('mold data only')

    #%%
    #################################
    if (not includeScans) and (not includeMoldScans):
        plt.show()
        return data
    else:
        #################################

        #%%
        print("Mapping scans with their corresponding expected mold...")

        def getMoldRowIdx(moldIdx):
            i = data.index[(data['type'] == 'mold') & (data['mold'] == moldIdx)].values
            if len(i) > 0:
                return i[0]
            else:
                return np.nan

        data['moldRow'] = [getMoldRowIdx(moldIdx) for moldIdx in list(data['mold'])]
        data = removeNanRows(data)

        data['moldRow'] = [int(i) for i in data['moldRow']] # convert to int

        print('testing moldRow')
        testi = data.index[min(700, data.values.shape[0]-1)]
        print('does {} == {}?'.format(
            data.loc[testi, 'mold'], # expected mold of scan 1000
            data.loc[data.loc[testi, 'moldRow'], 'mold'])) # the mold located at expected mold's row of scan 1000

        #%%
        print("Remove scans where expected mold is not available...")

        idxToKeep = []
        idxToRemove = []
        moldIdx = getMoldData().index
        for idx in data.index:
            if data.loc[idx, 'moldRow'] in moldIdx:
                idxToKeep.append(idx)
            else:
                idxToRemove.append(idx)
        data = data.loc[idxToKeep, :]

        print("Removing:")
        print(data[idxToRemove])

    return data
