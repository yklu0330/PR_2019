import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import random
import statistics as stat
import heapq

import Likelihood
import Bayes
import Header
import LinearClassifier
import PCA


def eigenface(trainData, testData, dataVariety):
    # standardize train data
    dropTrainData = trainData.drop("variety", axis=1)
    trainMean = dropTrainData.sum()
    trainMean = trainMean.values.reshape([dropTrainData.shape[1], 1])
    trainMean = trainMean / dropTrainData.shape[0]
    newtrainData = PCA.normalize(trainData, trainMean)

    # calculate xT * x and its eigenvector
    normTrainData = newtrainData.drop("variety", axis=1)
    normTrainData = np.array(normTrainData)
    X = np.transpose(normTrainData)

    tempMat = np.zeros([X.shape[1], X.shape[1]])
    np.matmul(np.transpose(X), X, tempMat)
    eigValX, eigVecX = np.linalg.eigh(tempMat)

    # calculate X * eigenvector
    newEigVecX = np.zeros([X.shape[0], eigVecX.shape[1]])
    newEigVecX = np.matmul(X, eigVecX)

    # normalize eigenvector
    newEigVecX = np.transpose(newEigVecX)
    length = np.linalg.norm(newEigVecX, axis=1)
    for i in range(newEigVecX.shape[0]):
        newEigVecX[i] /= length[i]
    normEigVec = np.transpose(newEigVecX)


    # calculate A
    L = 20
    maxEigIdx = np.argsort(-eigValX)
    A = []
    for i in range(L):
        A.append(normEigVec[:, maxEigIdx[i]])
    A = np.array(A)
    A = np.transpose(A)

    newtestData = PCA.normalize(testData, trainMean)

    # projection of train data
    projTrainFrame = PCA.project(A, newtrainData)

    # projection of test data
    projTestFrame = PCA.project(A, newtestData)

    # # classify test data by likelihood
    # g1, testIdx1, success1, confusion_mat1 = Likelihood.likelihood(projTrainFrame, projTestFrame, dataVariety)
    # Header.calAccuracy(success1, projTestFrame)
    # Header.ROC_AUC(projTestFrame, dataVariety, g1, testIdx1)
    # Header.drawConfusionMat(confusion_mat1, dataVariety)

    # classify test data by bayes
    names = []
    for i in range(projTestFrame.shape[1] - 1):
        names.append('0')
    names.append('variety')
    g2, testIdx2, success2, confusion_mat2 = Bayes.bayes(projTrainFrame, projTestFrame, dataVariety, names)
    Header.calAccuracy(success2, projTestFrame)
    Header.drawConfusionMat(confusion_mat2, dataVariety)

    # classify test data by linear classifier
    # g3, testIdx3, success3, confusion_mat3 = LinearClassifier.linearClassifier(projTrainFrame, projTestFrame, dataVariety)
    # Header.calAccuracy(success3, projTestFrame)
    # Header.ROC_AUC(projTestFrame, dataVariety, g3, testIdx3)
    # Header.drawConfusionMat(confusion_mat3, dataVariety)
