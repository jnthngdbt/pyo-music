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

#%% Some feature subsets. Select one to use.

featureSubset_ABCD = [ 'id', 'mold', 'specs-category', 'backA', 'backB', 'backC', 'backD', 'frontA', 'frontB', 'frontC', 'frontD', 'bottomA', 'bottomB', 'bottomC', 'bottomD', 'topA', 'topB', 'topC', 'topD', 'rightA', 'rightB', 'rightC', 'rightD', 'leftA', 'leftB', 'leftC', 'leftD']
featureSubset_ABCDK = [ 'id', 'mold', 'specs-category', 'backK', 'frontK', 'bottomK', 'topK', 'rightK', 'leftK', 'backA', 'backB', 'backC', 'backD', 'frontA', 'frontB', 'frontC', 'frontD', 'bottomA', 'bottomB', 'bottomC', 'bottomD', 'topA', 'topB', 'topC', 'topD', 'rightA', 'rightB', 'rightC', 'rightD', 'leftA', 'leftB', 'leftC', 'leftD']

# Selection
featureSubset = featureSubset_ABCDK

#%%

data = pd.read_csv('python/data/20190617.planes.molds.csv', index_col=False)

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

alpha = 0.5

def showSpecHists(specName, specCategories, featureName, bins):
    plt.figure(figsize=(8,3))

    plt.subplot(1,2,1)
    for spec in specCategories:
        d = data[data[specName] == spec]
        d = d[featureName]
        plt.hist(d, bins=bins, alpha=alpha)
        
    plt.xlabel(featureName)
    plt.legend(specCategories)

    plt.subplot(1,2,2)
    plt.hist(data[featureName], bins=bins, alpha=alpha)
    plt.xlabel(featureName)

    plt.suptitle(specName)
    plt.tight_layout()

showSpecHists('spec-back', faceNames, 'backK', np.arange(0, 0.05, 0.001))
showSpecHists('spec-front', faceNames, 'frontK', np.arange(0, 0.05, 0.001))
showSpecHists('spec-side', sideNames, 'leftK', np.arange(0, 0.01, 0.0002))
showSpecHists('spec-side', sideNames, 'rightK', np.arange(0, 0.01, 0.0002))

#%%

# def computeSpecBack():
#     # sel = data

#%%

plt.show()