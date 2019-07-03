#%%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# For this to work in VS Codeinteractive Python, the workspace root
# must be the folder of this file, for the folder structure to match.
import library
from library.initdata import *

# Selection
# featureSubset = featureSubset_ABCD
# featureSubset = featureSubset_Angle + featureSubset_Size + featureSubset_ABCDK
# featureSubset = featureSubset_Angle + featureSubset_Size + featureSubset_DK # should by alignment invariant
# featureSubset = featureSubset_Angle + featureSubset_Size + featureSubset_K # should by alignment invariant
featureSubset = featureSubset_Angle + featureSubset_DK # should by alignment invariant

#%%

data = importAndPreprocessData(
    featureSubset,
    includeScans=True, 
    includeMoldScans=True, 
    outlierScansStd=None, 
    subsampleScans=1, 
    showData=False,
    standardize=False)

def getMoldData(): return data.loc[data['type'] == 'mold', :]
def getScanData(): return data.loc[data['type'] == 'scan', :]
def getMoldScanData(): return data.loc[data['type'] == 'moldscan', :]
def getFullScanData(): return data.loc[(data['type'] == 'moldscan') | (data['type'] == 'scan'), :]

#%%

def computePerformance(yPredict, yTest, classesTrain):
    # yPredict: samples x classes
    Ns = yPredict.shape[0]

    plt.matshow(yPredict)

    plt.figure()
    ranks = 300*np.ones(Ns)
    for i in np.arange(Ns):
        if yTest[i] not in classesTrain:
            # a = 1
            print('expected {} not in trained classes'.format(yTest[i]))
        else:
            iexpected = classesTrain.index(yTest[i])

            sortIdx = np.argsort(yPredict[i,:])
            sortIdx = sortIdx[::-1] # descending
            rank = sortIdx.tolist().index(iexpected)
            ranks[i] = rank

            # # print(rank)
            # print('{} == {}'.format(classesTrain[iexpected], yTest[i]))
            # print('{} == {}'.format(yPredict[i,sortIdx[0]], np.max(yPredict[i,:])))
            # print('{} == {}'.format(yPredict[i, iexpected], yPredict[i, sortIdx[rank]]))

            # plt.plot(i, yPredict[i, sortIdx[0]], '.')
            plt.plot(rank, yPredict[i, sortIdx[0]], 'w.', alpha=0.5)
            plt.plot(rank, yPredict[i, iexpected], 'r.', alpha=0.5)
            # plt.xlim([0,100])
            # print('max: {}, sort: {}'.format(np.max(yPredict[i,:]), yPredict[i,sortIdx[0]]))

    plt.figure()
    plt.hist(ranks, bins=np.arange(0,100))

    def getPercent(sel):
        return 100.0 * np.sum(sel)/len(ranks)

    print('Top 1: {}%'.format(getPercent(ranks < 1)))
    print('Top 3: {}%'.format(getPercent(ranks < 3)))
    print('Top 5: {}%'.format(getPercent(ranks < 5)))
    print('Top 10: {}%'.format(getPercent(ranks < 10)))
    print('Top 20: {}%'.format(getPercent(ranks < 20)))


#%%

def computeLda(xTrain, xTest, yTrain, yTest):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as DA # 'pip install -U scikit-learn', or 'conda install scikit-learn'

    c = DA()
    c.fit(xTrain, yTrain) # samples x features
    # p = lda.predict_proba(xTest) # samples x classes
    yPredict = c.decision_function(xTest) # samples x classes

    computePerformance(yPredict, yTest, c.classes_.tolist())

def computeLogisticRegression(xTrain, xTest, yTrain, yTest):
    from sklearn.linear_model import LogisticRegression

    c = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
    c.fit(xTrain, yTrain) # samples x features
    # p = lda.predict_proba(xTest) # samples x classes
    yPredict = c.decision_function(xTest) # samples x classes

    computePerformance(yPredict, yTest, c.classes_.tolist())

#%% 

def testSplitFullScans():
    allData = getFullScanData()[featureSubset]
    allLabels = getFullScanData()['mold'].values

    from sklearn.model_selection import train_test_split
    return train_test_split(allData, allLabels, test_size = 0.2, random_state = 0)

def testCreaformOnMoldScans():
    xTrain = getMoldScanData()[featureSubset]
    yTrain = getMoldScanData()['mold'].values
    xTest = getMoldData()[featureSubset]
    yTest = getMoldData()['mold'].values
    return xTrain, xTest, yTrain, yTest 

def testScansOnMoldScans():
    xTrain = getMoldScanData()[featureSubset]
    yTrain = getMoldScanData()['mold'].values
    xTest = getScanData()[featureSubset]
    yTest = getScanData()['mold'].values
    return xTrain, xTest, yTrain, yTest 

def testMoldScansOnScans():
    xTrain = getScanData()[featureSubset]
    yTrain = getScanData()['mold'].values
    xTest = getMoldScanData()[featureSubset]
    yTest = getMoldScanData()['mold'].values
    return xTrain, xTest, yTrain, yTest 

# NOTE: cannot use creaform molds for training, since classes == samples

# xTrain, xTest, yTrain, yTest = testSplitFullScans()
# xTrain, xTest, yTrain, yTest = testCreaformOnMoldScans()
# xTrain, xTest, yTrain, yTest = testScansOnMoldScans()
xTrain, xTest, yTrain, yTest = testMoldScansOnScans()

print('Number for train: {} ({})'.format(len(yTrain), xTrain.shape[0]))
print('Number for test: {} ({})'.format(len(yTest), xTest.shape[0]))

# Why it is a good choice
# - inherently multiclass
# - no need to compute covariance; good since many classes with very few members
# - close form linear solution so easy to implement if we have to
# - there is a prior notion (a priori probability of a certain class), so built-in historical data
print('=== LDA ===')
computeLda(xTrain, xTest, yTrain, yTest)

# QDA: quadratic linear discriminant requires computing class covariance matrices. However,
# in our case classes have very few samples (often 1), so covariance is ill defined.

# # Hard to set parameters
# print('=== LOGISTIC REGRESSION ===')
# computeLogisticRegression(xTrain, xTest, yTrain, yTest)

#%%

plt.show()
