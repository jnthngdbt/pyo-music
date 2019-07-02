#%%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as DA # 'pip install -U scikit-learn', or 'conda install scikit-learn'

#----------------------------------
# # If running in interactive Python
# NOPE DOES NOT WORK
# # For this to work in interactive Python, set "Notebook File Root" setting to ${fileDirname}
# import initdata
# import initdata
#----------------------------------
# # If running in debugger.
import library
from library.initdata import *
#----------------------------------

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
featureSubset_simple = ['heightFront','lengthTop', 'slopeBack', 'widthTopFront', 'parallelismTop', 'parallelismDown', 'leftK', 'rightK']
featureSubset_simple4 = ['heightFront','slopeBack', 'widthTopFront', 'parallelismDown', 'leftK'] # lengthTop
featureSubset_simple2 = ['heightFront','lengthTop', 'slopeBack', 'widthTopFront', 'parallelismDown']
featureSubset_simple3 = ['heightFront','lengthTop', 'slopeBack']

# Selection
featureSubset = featureSubset_simple4

#%%

data = importAndPreprocessData(featureSubset, includeScans=True, includeMoldScans=False, outlierScansStd=3, subsampleScans=1, showData=False)

def getMoldData(): return data.loc[data['type'] == 'mold', :]
def getScanData(): return data.loc[data['type'] == 'scan', :]

#%%

def computeLdaWeights(data, classLabels):
    from sklearn.model_selection import train_test_split
    xTrain, xTest, yTrain, yTest = train_test_split(data, classLabels, test_size = 0.2, random_state = 0)

    lda = DA()
    lda.fit(xTrain, yTrain) # samples x features
    classesTrain = lda.classes_.tolist()

    # p = lda.predict_proba(xTest) # samples x classes
    p = lda.decision_function(xTest) # samples x classes

    Ns = p.shape[0]
    ranks = np.zeros(Ns)

    plt.matshow(p)

    plt.figure()
    for i in np.arange(Ns):
        if yTest[i] not in classesTrain:
            print('expected {} not in trained classes'.format(yTest[i]))
        else:
            iexpected = classesTrain.index(yTest[i])

            sortIdx = np.argsort(p[i,:])
            sortIdx = sortIdx[::-1] # descending
            rank = sortIdx.tolist().index(iexpected)
            ranks[i] = rank

            # print(rank)
            print('{} == {}'.format(classesTrain[iexpected], yTest[i]))
            print('{} == {}'.format(p[i,sortIdx[0]], np.max(p[i,:])))
            print('{} == {}'.format(p[i, iexpected], p[i, sortIdx[rank]]))

            # plt.plot(i, p[i, sortIdx[0]], '.')
            plt.plot(rank, p[i, sortIdx[0]], 'w.', alpha=0.5)
            plt.plot(rank, p[i, iexpected], 'r.', alpha=0.5)
            # plt.xlim([0,100])
            # print('max: {}, sort: {}'.format(np.max(p[i,:]), p[i,sortIdx[0]]))

    plt.figure()
    plt.hist(ranks, bins=np.arange(0,100))

    print('Top 3: {}%'.format(np.sum(ranks < 3)/len(ranks)))
    print('Top 5: {}%'.format(np.sum(ranks < 5)/len(ranks)))
    print('Top 10: {}%'.format(np.sum(ranks < 10)/len(ranks)))
    print('Top 20: {}%'.format(np.sum(ranks < 20)/len(ranks)))

#%% 

computeLdaWeights(getScanData()[featureSubset], getScanData()['mold'].values)

# -----------------------------------------------------------------------

plt.show()

theend = 0