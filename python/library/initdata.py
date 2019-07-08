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
featureSubset_ABCDK_NoTopNormal = [ 'backK', 'frontK', 'bottomK', 'topK', 'rightK', 'leftK', 'backA', 'backB', 'backC', 'backD', 'frontA', 'frontB', 'frontC', 'frontD', 'bottomA', 'bottomB', 'bottomC', 'bottomD', 'topD', 'rightA', 'rightB', 'rightC', 'rightD', 'leftA', 'leftB', 'leftC', 'leftD']
featureSubset_ABCDK_NoWeird = [ 'backK', 'frontK', 'bottomK', 'topK', 'rightK', 'leftK', 'backA', 'backB', 'backC', 'backD', 'frontA', 'frontB', 'frontC', 'frontD', 'bottomA', 'bottomB', 'bottomC', 'bottomD', 'topA', 'topB', 'topC', 'topD', 'rightA', 'rightB', 'rightC', 'rightD', 'leftA', 'leftB', 'leftC', 'leftD']
featureSubset_ABCK = [ 'backK', 'frontK', 'bottomK', 'topK', 'rightK', 'leftK', 'backA', 'backB', 'backC', 'frontA', 'frontB', 'frontC', 'bottomA', 'bottomB', 'bottomC', 'topA', 'topB', 'topC', 'rightA', 'rightB', 'rightC', 'leftA', 'leftB', 'leftC']
featureSubset_LdaTrialNope = [ 'topB', 'bottomB', 'bottomK', 'topK', 'rightK', 'leftK']
featureSubset_UncorrelateABCDK = [ 'backK', 'frontK', 'bottomK', 'topK', 'leftK', 'backA', 'backC', 'frontA', 'frontC', 'frontD', 'bottomB', 'bottomC', 'bottomD', 'topB', 'topD', 'rightA', 'rightB', 'rightD', 'leftA']
featureSubset_ABCD_NoTopBottomABC = [ 'backA', 'backB', 'backC', 'backD', 'frontA', 'frontB', 'frontC', 'frontD', 'bottomD', 'topD', 'rightA', 'rightB', 'rightC', 'rightD', 'leftA', 'leftB', 'leftC', 'leftD']
featureSubset_ABCD_NoTopBottomABC = [ 'backA', 'backB', 'backC', 'backD', 'frontA', 'frontB', 'frontC', 'frontD', 'bottomD', 'topD', 'rightA', 'rightB', 'rightC', 'rightD', 'leftA', 'leftB', 'leftC', 'leftD']

featureSubset_box1 = ['heightFront', 'heightBack', 'heightRatio', 'lengthTop', 'lengthDown', 'lengthRatio', 'slopeBack', 'slopeFront', 'slopeDown', 'slopeSide', 'parallelismTop', 'parallelismDown', 'parallelismRatio']
featureSubset_box2 = ['heightFront', 'heightBack', 'lengthTop', 'lengthDown', 'slopeBack', 'widthTopFront', 'widthTopBack', 'widthDownFront', 'widthDownBack', 'widthTopRatio', 'widthDownRatio', 'parallelismTop', 'parallelismDown', 'leftK', 'rightK']
featureSubset_box3 = ['heightFront', 'heightBack', 'lengthTop', 'lengthDown', 'slopeBack', 'parallelismTop', 'parallelismDown', 'frontK', 'front-planarity', 'backK', 'back-planarity', 'leftK', 'left-planarity', 'rightK', 'right-planarity']

featureSubset_Size = ['heightFront', 'heightBack', 'heightRatio', 'lengthTop', 'lengthDown', 'lengthRatio', 'widthTopFront', 'widthTopBack', 'widthDownFront', 'widthDownBack', 'widthTopRatio', 'widthDownRatio']
featureSubset_SizeConcise = ['heightFront', 'lengthTop', 'widthDownFront', 'widthDownBack']
featureSubset_Angle = ['slopeBack', 'slopeFront', 'slopeDown', 'slopeSide', 'parallelismTop', 'parallelismDown', 'parallelismRatio']
featureSubset_AngleConcise = ['slopeBack', 'slopeFront', 'slopeDown', 'slopeSide', 'parallelismDown']

def removeNanRows(df):
    nanRows = df.isna().any(axis=1)

    print('removing following NaN rows:')
    print(df[nanRows == True])

    return df[nanRows != True]

from scipy import stats
def removeOutliers(df, featureSubset, outlierStd):
    if outlierStd != None:
        inlierMask = (np.abs(stats.zscore(df[featureSubset])) < outlierStd).all(axis=1)
        outlierMask = np.logical_not(inlierMask)
        print(df[outlierMask])
        df = df[inlierMask]
    return df

def rereferenceNormals(data):
    print("Re-referencing normal features...")

    def normalize(n):
        return n / np.linalg.norm(n)

    def getNormal(data, i, name):
        n = data.iloc[i,:]
        n = n[[name + 'A', name + 'B', name + 'C']].astype('float64')
        n = n.values.flatten()
        return normalize(n)

    def setTransformedNormal(n, data, i, name, T):
        n = normalize(np.dot(T, n))
        idx = data.index[i]
        data.at[idx, name + 'A'] = n[0]
        data.at[idx, name + 'B'] = n[1]
        data.at[idx, name + 'C'] = n[2]

    for i in np.arange(data.shape[0]):
        top = getNormal(data, i, 'top')
        down = getNormal(data, i, 'bottom')
        left = getNormal(data, i, 'left')
        right = getNormal(data, i, 'right')
        front = getNormal(data, i, 'front')
        back = getNormal(data, i, 'back')

        # Construct basis
        y = normalize(-top)
        xe = front + back # NOTE: PROBABLY A BUG IN PLANES FEATURES, BOTH FRONT AND BACK X ARE POSITIVE, BUT NOT ALWAYS
        xe[1] = 0 # only keep x-z components
        xe = normalize(xe)
        z = normalize(np.cross(xe, y))
        x = normalize(np.cross(y, z))
        T = np.array([x,y,z])

        setTransformedNormal(top, data, i, 'top', T)
        setTransformedNormal(down, data, i, 'bottom', T)
        setTransformedNormal(left, data, i, 'left', T)
        setTransformedNormal(right, data, i, 'right', T)
        setTransformedNormal(front, data, i, 'front', T)
        setTransformedNormal(back, data, i, 'back', T)

    ss = [ 'backA', 'backB', 'backC', 'frontA', 'frontB', 'frontC', 'bottomA', 'bottomB', 'bottomC', 'topA', 'topB', 'topC']
    print(data[ss])
    return data

def importAndPreprocessData(
    featureSubset, 
    moldsFile=None, 
    scansFiles=None, 
    moldScansFiles=None, 
    subsampleScans=3, 
    outlierScansStd=None, 
    outlierMoldsStd=None,
    computeMoldRowMapping=True,
    ignoreBfi=False,
    showData=False,
    rereferenceNormalFeatures=False,
    standardize=False):

    print("Creating main database...")

    # Merge scans and molds
    moldData = pd.read_csv(moldsFile, index_col=False)
    moldData = removeNanRows(moldData)
    if rereferenceNormalFeatures: # do it before removing outliers
        moldData = rereferenceNormals(moldData)
    moldData = removeOutliers(moldData, featureSubset, outlierMoldsStd)
    moldData['type'] = 'mold'
    data = moldData

    if scansFiles != None:
        for f in scansFiles:
            scanData = pd.read_csv(f, index_col=False)
            scanData = removeNanRows(scanData)
            if rereferenceNormalFeatures: # do it before removing outliers
                scanData = rereferenceNormals(scanData)
            scanData = removeOutliers(scanData, featureSubset, outlierScansStd)
            scanData['type'] = 'scan'
            # Subsample
            scanData = scanData.iloc[np.arange(0, scanData.shape[0], subsampleScans), :]
            data = data.append(scanData, ignore_index=True)

    if moldScansFiles != None:
        for f in moldScansFiles:
            moldScanData = pd.read_csv(f, index_col=False)
            moldScanData = removeNanRows(moldScanData)
            if rereferenceNormalFeatures:
                scanData = rereferenceNormals(scanData)
            moldScanData['type'] = 'moldscan'
            data = data.append(moldScanData, ignore_index=True)

    def getMoldData(): return data.loc[data['type'] == 'mold', :]
    def getScanData(): return data.loc[data['type'] == 'scan', :]
    def getMoldScanData(): return data.loc[data['type'] == 'moldscan', :]
    def getFullScanData(): return data.loc[(data['type'] == 'moldscan') | (data['type'] == 'scan'), :]

    print('appended into a single dataframe')

    #%%
    if ignoreBfi:
        print("Removing BFI scans...")
        data = data[data['id'] < 6000]

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
    if (standardize):
        print("Standardize features...")

        for feat in featureSubset:
            data[feat] = (data[feat] - data[feat].mean(axis=0)) / data[feat].std(axis=0)

    #%%
    print("Showing mold priors...")

    data[['mold', 'type']].groupby('type').hist(bins=np.arange(data['mold'].max()))

    #%%
    if showData:
        print("Showing data with pandas visualization...")

        scanData = getFullScanData()
        moldData = getMoldData()

        plt.figure(figsize=(10,10))
        Nf = len(featureSubset)
        Nc = 5
        Nr = np.ceil(Nf / Nc)
        for i in np.arange(Nf):
            plt.subplot(Nr, Nc, i+1)
            plt.hist(scanData[featureSubset[i]], bins=50, alpha=0.5)
            plt.hist(moldData[featureSubset[i]], bins=50, alpha=0.5)
            plt.xlabel(featureSubset[i])
        plt.suptitle('scan then mold')
        plt.tight_layout()

    #%%
        if (len(featureSubset) < 10): 
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
    if (scansFiles == None) and (moldScansFiles == None):
        plt.show()
        return data
    else:
        #################################

        if computeMoldRowMapping:
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
