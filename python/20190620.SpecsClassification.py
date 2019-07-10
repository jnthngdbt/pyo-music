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

data = pd.read_csv('data/planes.molds.csv', index_col=False)

def removeNanRows(df):
    nanRows = df.isna().any(axis=1)

    print('removing following NaN rows:')
    print(df[nanRows == True])

    return df[nanRows != True]

data = removeNanRows(data)
data = data[data['id'] > 0] # Remove id 0, which is the scan when generating the molds data.

print('preprocessed data:')
print(data.head())

#%%

data['type'] = 'mold'

sideNames = ['parallel', 'bowed', 'tapered']
faceNames = ['n/a', 'square', 'semiround', 'n/a', 'round'] # (front, back) bit mask, so power of 2
lengthNames = ['four', 'fourhalf', 'five', 'fivehalf', 'six']

data['spec-side'] = [sideNames[i] for i in data['eside']]
data['spec-back'] = [faceNames[i] for i in data['eback']]
data['spec-front'] = [faceNames[i] for i in data['efront']]
data['spec-length'] = [lengthNames[i] for i in data['elength']]

faceNames = ['square', 'semiround', 'round'] # remove the power of 2 thing

print('added categories:')
print(data.head())

#%%

def getCentroid(plane, i):
    c = data.loc[i, [plane+'-cx', plane+'-cy', plane+'-cz']].values
    return c

def getNormWithoutComponent(v, i):
    v[i] = 0
    return np.linalg.norm(v)

data['my-spec-length'] = np.abs(data['front-cx']) + np.abs(data['back-cx'])
data['my-spec-height'] = np.abs(data['bottom-cy']) + np.abs(data['top-cy'])

#%%

def getNormal(plane, i):
    n = data.loc[i, [plane+'A', plane+'B', plane+'C']].values
    n = n / np.linalg.norm(n)
    return n

def getNormalForParallelism(plane, i):
    n = getNormal(plane, i)
    n[1] = 0 # kill y component
    n = n / np.linalg.norm(n)
    return n

def convertDotToAngleDeg(dot):
    a = 180.0 * np.arccos(dot) / np.pi
    return a

def computeParallelism(plane1, plane2, i):
    n1 = getNormalForParallelism(plane1, i)
    n2 = getNormalForParallelism(plane2, i)
    d = convertDotToAngleDeg(min(1.0, np.dot(n1, n2)))
    return d

data['parallelism'] = [computeParallelism('left', 'right', i) for i in data.index]

data = removeNanRows(data)

#%%

alpha = 0.5

def showSpecHists(data, specName, specCategories, featureName, bins, ylim = None):
    plt.figure(figsize=(8,3))

    if specName != None:
        plt.subplot(1,2,1)
        for spec in specCategories:
            d = data[data[specName] == spec]
            d = d[featureName]
            plt.hist(d, bins=bins, alpha=alpha)
            
        plt.xlabel(featureName)
        plt.legend(specCategories)
        plt.suptitle(specName)
        if ylim != None: plt.ylim(ylim)

    plt.subplot(1,2,2)
    plt.hist(data[featureName], bins=bins, alpha=alpha)
    plt.xlabel(featureName)
    if ylim != None: plt.ylim(ylim)

    plt.tight_layout()

showSpecHists(data, 'spec-back', faceNames, 'backK', np.arange(0, 0.05, 0.001))
showSpecHists(data, 'spec-front', faceNames, 'frontK', np.arange(0, 0.05, 0.001))
# showSpecHists(data, 'spec-side', sideNames, 'leftK', np.arange(0, 0.01, 0.0002))
# showSpecHists(data, 'spec-side', sideNames, 'rightK', np.arange(0, 0.01, 0.0002))
showSpecHists(data, 'spec-length', lengthNames, 'my-spec-length', np.arange(0.75, 1.5, 0.02))
showSpecHists(data, 'spec-side', sideNames, 'parallelism', np.arange(0, 7, 0.2), [0,20])

#%%

plt.show()