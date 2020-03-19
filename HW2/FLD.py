import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import random

def SwSb(dataset, dataVariety):
    mean = np.zeros((len(dataVariety), dataset.shape[1] - 1, 1))
    cov = np.zeros((len(dataVariety), dataset.shape[1] - 1, dataset.shape[1] - 1))

    prob = []
    Sw = np.zeros((dataset.shape[1] - 1, dataset.shape[1] - 1))
    Sb = np.zeros((dataset.shape[1] - 1, dataset.shape[1] - 1))

    # calculate Sw
    for i in range(len(dataVariety)):
        classData = dataset[dataset['variety'] == dataVariety[i]]
        newData = classData.drop("variety", axis=1)

        mu = newData.sum()
        mu = mu.values.reshape([newData.shape[1], 1])
        mu = mu / newData.shape[0]
        mean[i] = mu

        for index, row in newData.iterrows():
            covar = np.zeros([newData.shape[1], newData.shape[1]])
            xk = row.values.reshape((newData.shape[1], 1))
            vector = xk - mu

            np.matmul(vector, np.transpose(vector), covar)
            cov[i] += covar

        cov[i] = cov[i] / newData.shape[0]

        prob.append(classData.shape[0] / dataset.shape[0])

    for i in range(len(dataVariety)):
        Sw += prob[i] * cov[i]

    Sw /= len(dataVariety)

    # calculate Sb
    mean0 = np.zeros((dataset.shape[1] - 1, 1))
    for i in range(len(dataVariety)):
        mean0 += prob[i] * mean[i]

    mat = np.zeros((dataset.shape[1] - 1, dataset.shape[1] - 1))
    for i in range(len(dataVariety)):
        vector = mean[i] - mean0
        np.matmul(vector, np.transpose(vector), mat)
        Sb += prob[i] * mat
    return mean, Sw, Sb

def FLD(dataset, dataVariety):
    # randomize data
    randomData = dataset.sample(frac=1)
    datasize = randomData.shape[0]

    # split the train data and test data
    trainData = randomData[:int(datasize * 0.75)]
    testData = randomData[int(datasize * 0.75):]

    mean, Sw, Sb = SwSb(trainData, dataVariety)

    # calculate w
    w = np.zeros((trainData.shape[1] - 1, 1))
    vector = mean[0] - mean[1]
    np.matmul(np.linalg.inv(Sw), vector, w)

    # calculate J before projection
    mean1, Sw1, Sb1 = SwSb(dataset, dataVariety)
    matTemp1 = np.zeros((dataset.shape[1] - 1, dataset.shape[1] - 1))
    np.matmul(np.linalg.inv(Sw1), Sb1, matTemp1)
    J1 = np.trace(matTemp1)
    print("separability measures before the projection: %f" %J1)

    # projection of x
    G = []
    testIdX = []
    dataVar = []
    for index, row in dataset.iterrows():
        arr = np.array(row[:dataset.shape[1] - 1])
        x = arr.reshape((dataset.shape[1] - 1, 1))
        g = np.dot(np.transpose(w), x)
        G.append(g[0][0])
        testIdX.append(index)
        dataVar.append(row[dataset.shape[1] - 1])

    Gframe = pd.DataFrame(G, columns=['gValue'])
    varFrame = pd.DataFrame(dataVar, columns=['variety'])
    newFrame = pd.concat([Gframe, varFrame], axis=1, join_axes=[Gframe.index])

    # calculate J after projection
    mean2, Sw2, Sb2 = SwSb(newFrame, dataVariety)
    matTemp2 = np.zeros((newFrame.shape[1] - 1, newFrame.shape[1] - 1))
    np.matmul(np.linalg.inv(Sw2), Sb2, matTemp2)
    J2 = np.trace(matTemp2)
    print("separability measures after the projection: %f" % J2)

    return G, testIdX
