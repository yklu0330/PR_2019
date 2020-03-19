import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def addNoise(dataset):
    dropData = dataset.drop("variety", axis=1)
    dataArr = np.array(dropData)

    temp = []
    noise = []
    for i in range(dropData.shape[1]):
        for j in range(dropData.shape[0]):
            temp.append(dataArr[j][i])
        noise.append(np.random.uniform(min(temp), max(temp), int(dataset.shape[0] * 0.1)))
        temp = []
    noisePoint = np.array(noise)
    noisePoint = np.transpose(noisePoint)

    noisePoint = pd.DataFrame(noisePoint, columns=dropData.columns)
    dataVar = []
    for i in range(noisePoint.shape[0]):
        dataVar.append('noise')
    dataVarFrame = pd.DataFrame(dataVar, columns=['variety'])

    noiseFrame = pd.concat([noisePoint, dataVarFrame], axis=1)
    newFrame = pd.concat([dataset, noiseFrame], axis=0, ignore_index=True)

    return newFrame

# load iris dataset from csv
def loadIris(noise):
    names = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width', 'variety']
    dataset = pd.read_csv('./iris.csv', names=['sepal.length', 'sepal.width', 'petal.length', 'petal.width', 'variety'], skiprows=0)

    if noise == 1:
        dataset = addNoise(dataset)

    randomData = dataset.sample(frac=1)
    dataVariety = randomData['variety'].unique()
    datasize = randomData.shape[0]

    return randomData, dataVariety, datasize

# load wine dataset from csv
def loadWine(noise):
    names=['variety', 'alco', 'malic', 'ash', 'alcal', 'mag', 'total', 'flav', 'nonflav', 'proan', 'color', 'hue', 'OD', 'proline']
    dataset = pd.read_csv('./wine.csv', names=['variety', 'alco', 'malic', 'ash', 'alcal', 'mag', 'total',
                                                   'flav', 'nonflav', 'proan', 'color', 'hue', 'OD', 'proline'], skiprows=0)

    if noise == 1:
        dataset = addNoise(dataset)

    randomData = dataset.sample(frac=1)
    dataVariety = dataset['variety'].unique()
    datasize = randomData.shape[0]

    return randomData, dataVariety, datasize

# load ionos dataset from csv
def loadIonos(noise):
    names = ['c', 'd', 'e', 'f', 'g',
             'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
             'u', 'v', 'w', 'x', 'y', 'z', 'ab', 'bc', 'cd', 'de', 'ef',
             'fg', 'gh', 'hi', 'variety']
    dataset = pd.read_csv('./ionosphere.csv', names=['c', 'd', 'e', 'f', 'g',
                                                     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                                                     'u', 'v', 'w', 'x', 'y', 'z', 'ab', 'bc', 'cd', 'de', 'ef',
                                                     'fg', 'gh', 'hi', 'variety'], skiprows=0)

    if noise == 1:
        dataset = addNoise(dataset)

    randomData = dataset.sample(frac=1)
    dataVariety = dataset['variety'].unique()
    datasize = randomData.shape[0]

    return randomData, dataVariety, datasize

# load vertebral dataset from csv
def loadVertebral(noise):
    names = ['a', 'b', 'c', 'd', 'e', 'f', 'variety']
    dataset = pd.read_csv('./vertebral.csv', names=['a', 'b', 'c', 'd', 'e', 'f', 'variety'], skiprows=0)

    if noise == 1:
        dataset = addNoise(dataset)

    randomData = dataset.sample(frac=1)
    dataVariety = dataset['variety'].unique()
    datasize = randomData.shape[0]

    return randomData, dataVariety, datasize

def calEucDist(vecX, vecY):
    return np.sqrt(np.sum(np.power(vecX - vecY, 2)))

def clustSorting(dataArr, dataLabel):
    dataVariety = np.unique(dataLabel)

    clust = []
    for i in range(len(dataVariety)):
        index = np.argwhere(dataLabel == dataVariety[i])
        clusterData = np.zeros([index.shape[0], dataArr.shape[1]])
        for j in range(index.shape[0]):
            clusterData[j] = dataArr[index[j]]
        clust.append(clusterData)

    x_mean = np.zeros([len(dataVariety)])
    for i in range(len(dataVariety)):
        x = clust[i][:, 0]
        x_mean[i] = np.mean(x)

    sortClust = []
    sortX_mean = np.argsort(x_mean)
    sortLabel = np.zeros([dataLabel.shape[0]])
    for i in range(dataLabel.shape[0]):
        temp = np.argwhere(dataVariety == dataLabel[i])
        sortLabel[i] = np.argwhere(sortX_mean == temp[0])

    for i in range(len(dataVariety)):
        sortClust.append(clust[sortX_mean[i]])
    return sortClust, sortLabel

def plotFig(clust, filename):
    color = ['ro', 'yo', 'bo']
    for i in range(len(clust)):
        # iris: 0, 1
        # wine: 6, 7, 8, 9, 10, 12 (6, 12)
        x = clust[i][:, 0]
        x = x.tolist()
        y = clust[i][:, 1]
        y = y.tolist()
        plt.plot(x, y, color[i])
    plt.draw()
    plt.savefig(filename)
    plt.show()

def plotNoiseFig(clust, filename, noiseArr):
    color = ['ro', 'yo', 'bo']

    for i in range(len(clust)):
        # iris: 0, 1
        # wine: 6, 7, 8, 9, 10, 12 (6, 12)
        x = clust[i][:, 0]
        x = x.tolist()
        y = clust[i][:, 1]
        y = y.tolist()
        plt.plot(x, y, color[i])

    noiseX = noiseArr[:, 0]
    noiseX = noiseX.tolist()
    noiseY = noiseArr[:, 1]
    noiseY = noiseY.tolist()
    plt.plot(noiseX, noiseY, 'ko')

    plt.draw()
    plt.savefig(filename)
    plt.show()

def realDataSorting(dataArr, dataLabel, dataVariety, noise):
    if noise == 1:
        # find noise data
        noiseIdx = np.argwhere(dataLabel == 'noise')
        noiseId = []
        for i in range(noiseIdx.shape[0]):
            noiseId.append(noiseIdx[i][0])
        noiseList = []
        for i in noiseId:
            noiseList.append(dataArr[i])
        noiseArr = np.array(noiseList)

    # cluster the normal data
    if noise == 1:
        idx = np.argwhere(dataVariety == 'noise')
        dataVariety = np.delete(dataVariety, idx)
        for i in noiseId:
            varIdx = np.random.randint(0, 3, dtype='int')
            dataLabel[i] = dataVariety[varIdx]

    clust = []
    for i in range(len(dataVariety)):
        index = np.argwhere(dataLabel == dataVariety[i])
        clusterData = np.zeros([index.shape[0], dataArr.shape[1]])
        for j in range(index.shape[0]):
            clusterData[j] = dataArr[index[j][0]]
        clust.append(clusterData)

    x_mean = np.zeros([len(dataVariety)])
    for i in range(len(dataVariety)):
        x = clust[i][:, 0]
        x_mean[i] = np.mean(x)

    sortClust = []
    sortX_mean = np.argsort(x_mean)
    sortLabel = np.zeros([dataLabel.shape[0]])
    for i in range(dataLabel.shape[0]):
        temp = np.argwhere(dataVariety == dataLabel[i])
        sortLabel[i] = np.argwhere(sortX_mean == temp[0])

    for i in range(len(dataVariety)):
        sortClust.append(clust[sortX_mean[i]])

    if noise == 1:
        return sortClust, sortLabel, noiseArr
    else:
        return sortClust, sortLabel