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
# featureSubset = featureSubset_ABC
# featureSubset = featureSubset_ABCD
featureSubset = featureSubset_ABCDK
# featureSubset = featureSubset_ABCDK_NoTop
# featureSubset = featureSubset_ABCDK_NoTopNormal
# featureSubset = featureSubset_ABCK + featureSubset_Size
# featureSubset = featureSubset_ABCK + featureSubset_Size
# featureSubset = featureSubset_Angle + featureSubset_Size + featureSubset_ABCDK
# featureSubset = featureSubset_Angle + featureSubset_Size + featureSubset_DK # should by alignment invariant
# featureSubset = featureSubset_AngleConcise + featureSubset_SizeConcise + featureSubset_K # should by alignment invariant
# featureSubset = featureSubset_Angle + featureSubset_Size + featureSubset_K # should by alignment invariant
# featureSubset = featureSubset_Angle + featureSubset_DK # should by alignment invariant
# featureSubset = featureSubset_DK # should by alignment invariant
# featureSubset = featureSubset_D # should by alignment invariant
# featureSubset = featureSubset_K # should by alignment invariant
# featureSubset = featureSubset_Edges + featureSubset_K
# featureSubset = featureSubset_DK
# featureSubset = featureSubset_Edges + featureSubset_ABCDK

#%%

# NOTES
# - allscans contain moldscans and goodscans
# - standardizing does not change results

data = importAndPreprocessData(
    featureSubset,
    # moldsFile='data/20190617.planes.molds.csv', 
    # scansFiles=['data/20190617.planes.moldscans.csv'], 
    # scansFiles=['data/20190617.planes.allscans.csv'], 
    # scansFiles=['data/20190703.planes.bfiscans.csv', 'data/20190617.planes.moldscans.csv'], 
    # scansFiles=['data/20190703.planes.bfiscans.csv', 'data/20190617.planes.allscans.csv'], 
    # scansFiles=['data/20190703.planes.bfiscans.csv', 'data/20190703.planes.bfiscans.rawalign.csv'], 
    # scansFiles=['data/planes.bfiscans.rawalign.csv'], 
    # moldScansFiles=['data/20190703.planes.bfiscans.csv'], 
    # moldScansFiles=['data/20190617.planes.goodscans.csv'], 
    # moldScansFiles=['data/20190703.planes.bfiscans.csv', 'data/20190617.planes.allscans.csv'], 
    # moldScansFiles=['data/20190703.planes.goodscans.rawalign.csv'], 
    moldsFile='data/planes.moldscans.ori.rawalign.csv', 
    scansFiles=['data/planes.bfiscans.ori.rawalign.csv'], 
    outlierMoldsStd=4, 
    outlierScansStd=3, 
    ignoreBfi=False,
    subsampleScans=1, 
    showData=False,
    computeMoldRowMapping=False,
    # rereferenceNormalFeatures=True,
    standardize=False) # does not change results, but may help for visualization

reduceWithPca = False

def getMoldData(): return data.loc[data['type'] == 'mold', :]
def getScanData(): return data.loc[data['type'] == 'scan', :]
def getMoldScanData(): return data.loc[data['type'] == 'moldscan', :]
def getFullScanData(): return data.loc[(data['type'] == 'moldscan') | (data['type'] == 'scan'), :]

#%%

def computePerformance(c, yPredict, yTest, classesTrain):
    from sklearn.metrics import accuracy_score
    print('Accuracy: {}%'.format(100.0 * accuracy_score(yTest, c.predict(xTest))))

    # yPredict: samples x classes
    Ns = yPredict.shape[0]

    plt.figure()
    for i in np.arange(yPredict.shape[0]):
        plt.plot(np.sort(yPredict[i,:]) - np.max(yPredict[i,:]))
    plt.xlabel('class')
    plt.ylabel('hyper-distance')

    yPredictBest = []

    nbConsidered = 0;

    plt.figure()
    ranks = 300*np.ones(Ns)
    for i in np.arange(Ns):
        if yTest[i] not in classesTrain:
            a = 1
            # print('expected {} not in trained classes'.format(yTest[i]))
        else:
            nbConsidered += 1
            iexpected = classesTrain.index(yTest[i])

            sortIdx = np.argsort(yPredict[i,:])
            sortIdx = sortIdx[::-1] # descending
            rank = sortIdx.tolist().index(iexpected)
            ranks[i] = rank

            yPredictBest.append(classesTrain[sortIdx[0]])

            # # print(rank)
            # print('{} == {}'.format(classesTrain[iexpected], yTest[i]))
            # print('{} == {}'.format(yPredict[i,sortIdx[0]], np.max(yPredict[i,:])))
            # print('{} == {}'.format(yPredict[i, iexpected], yPredict[i, sortIdx[rank]]))

            plt.plot(rank, yPredict[i, sortIdx[0]], 'w.', alpha=0.5)
            plt.plot(rank, yPredict[i, iexpected], 'r.', alpha=0.5)

    plt.xlabel('rank')
    plt.ylabel('score (signed distance to hyperplane)')
    plt.legend(['best', 'expected'])

    plt.figure()
    plt.hist(ranks, bins=np.arange(0,100))
    plt.xlabel('rank')

    plt.figure()
    plt.hist(yTest, bins=np.arange(0,600))
    plt.xlabel('expected mold')

    plt.figure()
    plt.hist(yPredictBest, bins=np.arange(0,600))
    plt.xlabel('predicted mold')

    def getPercent(sel):
        return '{} ({})'.format(100.0 * np.sum(sel)/len(ranks), 100.0 * np.sum(sel)/nbConsidered)

    print('Top 1: {}%'.format(getPercent(ranks < 1)))
    print('Top 3: {}%'.format(getPercent(ranks < 3)))
    print('Top 5: {}%'.format(getPercent(ranks < 5)))
    print('Top 10: {}%'.format(getPercent(ranks < 10)))
    print('Top 20: {}%'.format(getPercent(ranks < 20)))
    print('Top 30: {}%'.format(getPercent(ranks < 30)))


#%%

def printLdaWeights(lda, classesTrain):
    with open('out.txt', 'w') as file:
        weights = lda.coef_
        nbClasses = weights.shape[0]
        nbFeatures = weights.shape[1]

        # Print header
        hdrStr = 'mold,'
        if reduceWithPca: 
            for i in np.arange(nbFeatures): hdrStr += 'PC{},'.format(i)
        else: 
            for f in featureSubset: hdrStr += f + ','
        file.write(hdrStr + '\n')

        for i in np.arange(nbClasses):
            rowStr = '{0:4d},'.format(classesTrain[i])
            for j in np.arange(nbFeatures):
                rowStr += '{},'.format(weights[i,j])
            file.write(rowStr + '\n')

def getPca(data, varianceRatio):
    from sklearn.decomposition import PCA
    # From the docs: "The input data is centered but not scaled for each feature before applying the SVD."
    pca = PCA(n_components=varianceRatio, svd_solver='full') 
    pca.fit(data)
    return pca

def computeLda(xTrain, xTest, yTrain, yTest):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as DA # 'pip install -U scikit-learn', or 'conda install scikit-learn'

    c = DA()
    c.fit(xTrain, yTrain) # samples x features
    # yPredict = c.predict_proba(xTest) # samples x classes
    yPredict = c.decision_function(xTest) # samples x classes

    classNames = c.classes_.tolist()
    printLdaWeights(c, classNames)

    if not reduceWithPca:
        weights = c.coef_
        plt.figure(figsize=(10,10))
        ax = plt.subplot(1,1,1)
        ax.matshow(weights)
        ax.set_aspect('auto')
        ax.set_xticks(np.arange(len(featureSubset)))
        ax.set_yticks(np.arange(len(classNames)))
        ax.set_xticklabels(featureSubset, rotation=90, fontsize=10)
        ax.set_yticklabels(classNames, fontsize=6)

    computePerformance(c, yPredict, yTest, classNames)

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

def testSplitScans():
    allData = getScanData()[featureSubset]
    allLabels = getScanData()['mold'].values

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

def testScansOnMolds():
    xTrain = getMoldData()[featureSubset]
    yTrain = getMoldData()['mold'].values
    xTest = getScanData()[featureSubset]
    yTest = getScanData()['mold'].values
    return xTrain, xTest, yTrain, yTest 

# NOTE: cannot use creaform molds for training, since classes == samples

# xTrain, xTest, yTrain, yTest = testSplitFullScans()
# xTrain, xTest, yTrain, yTest = testSplitScans()
# xTrain, xTest, yTrain, yTest = testCreaformOnMoldScans()
# xTrain, xTest, yTrain, yTest = testScansOnMoldScans()
# xTrain, xTest, yTrain, yTest = testMoldScansOnScans()
xTrain, xTest, yTrain, yTest = testScansOnMolds()

if reduceWithPca:
    varianceRatio = 0.99
    pca = getPca(xTrain, varianceRatio)
    print('PCA reduction from {} to {} dimensions ({}%)'.format(xTrain.shape[1], pca.transform(xTrain).shape[1], varianceRatio*100.0))
    xTrain = pca.transform(xTrain)
    xTest = pca.transform(xTest)

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
