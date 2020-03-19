import numpy as np
import pandas as pd
import statistics as stat

import Header

def normalize(data, mu):
    dataVar = np.array(data['variety'])
    dataVar = dataVar.reshape([data.shape[0], 1])
    dropData = data.drop("variety", axis=1)

    var = []
    for i in range(dropData.shape[1]):
        list = []
        list = dropData.ix[:, i]
        var.append(stat.variance(list))

    arrData = np.array(dropData)
    for i in range(arrData.shape[0]):
        for j in range(arrData.shape[1]):
            arrData[i][j] -= mu[j]
            arrData[i][j] /= var[j]
    DataF = pd.DataFrame(arrData)

    dataVarFrame = pd.DataFrame(dataVar, columns=['variety'])
    normFrame = pd.concat([DataF, dataVarFrame], axis=1)

    return normFrame

def calCov(data):
    data = pd.DataFrame(data)

    mu = data.sum()
    mu = mu.values.reshape([data.shape[1], 1])
    mu = mu / data.shape[0]

    cov = np.zeros([data.shape[1], data.shape[1]])

    for index, row in data.iterrows():
        xk = row.values.reshape((data.shape[1], 1))
        vector = xk - mu

        for i in range(data.shape[1]):
            for j in range(data.shape[1]):
                cov[i][j] += vector[i] * vector[j]

    cov = cov / data.shape[0]

    return cov

def project(A, data):
    # projection of data
    dataVar = np.array(data['variety'])
    dataVar = dataVar.reshape([data.shape[0], 1])

    newData1 = data.drop("variety", axis=1)
    newData1 = np.array(newData1)
    newData1 = np.transpose(newData1)
    projData = np.zeros([A.shape[1], newData1.shape[1]])
    np.matmul(np.transpose(A), newData1, projData)

    projData = np.transpose(projData)
    projDF = pd.DataFrame(projData)

    dataVarFrame = pd.DataFrame(dataVar, columns=['variety'])
    newFrame = pd.concat([projDF, dataVarFrame], axis=1)

    return newFrame

def PCA(dataset):
    # standardize data
    dropData = dataset.drop("variety", axis=1)
    dataMean = dropData.sum()
    dataMean = dataMean.values.reshape([dropData.shape[1], 1])
    dataMean = dataMean / dropData.shape[0]
    newData = normalize(dataset, dataMean)

    # calculate covX
    dropData2 = newData.drop("variety", axis=1)
    covX = calCov(dropData2)

    # calculate A
    eigValX, eigVecX = np.linalg.eig(covX)
    L = 2
    maxEigIdx = np.argsort(-eigValX)
    A = []
    for i in range(L):
        A.append(eigVecX[:, maxEigIdx[i]])
    A = np.array(A)
    A = np.transpose(A)

    # projection of data
    projFrame = project(A, newData)

    return projFrame
