import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import statistics as stat
import random
from scipy.stats import norm
from scipy.stats import multivariate_normal
import math

def maxLikelihood(data):
    data = pd.DataFrame(data)
    newData = data.drop("variety", axis=1)

    mu = newData.sum()
    mu = mu.values.reshape([newData.shape[1], 1])
    mu = mu / newData.shape[0]

    cov = np.zeros([newData.shape[1], newData.shape[1]])

    for index, row in newData.iterrows():
        xk = row.values.reshape((newData.shape[1], 1))
        vector = xk - mu

        for i in range(newData.shape[1]):
            for j in range(newData.shape[1]):
                cov[i][j] += vector[i] * vector[j]

    cov = cov / newData.shape[0]

    return mu, cov

def gi(data, mu, cov, P):
    data = pd.DataFrame(data)
    newData = data.drop("variety", axis=0)

    x = np.zeros([newData.shape[0], 1])
    for i in range(len(x)):
        x[i] = newData.ix[i]


    detCov = np.linalg.det(cov)

    expPow = -1 / 2 * np.dot(np.transpose(x - mu), np.dot(np.linalg.inv(cov), x - mu))
    div = math.pow(2 * math.pi, len(dataVariety)) * abs(detCov)
    pdfVal = 1 / math.sqrt(div) * math.exp(expPow)

    gi = pdfVal * P
    # gi = math.log(pdfVal * P)

    return gi


# load iris dataset from csv
dataset = pd.read_csv('./iris.csv', names=['sepal.length', 'sepal.width', 'petal.length', 'petal.width', 'variety'], skiprows=0)

# # load wine dataset from csv
# names=['variety', 'alco', 'malic', 'ash', 'alcal', 'mag', 'total', 'flav', 'nonflav', 'proan', 'color', 'hue', 'OD', 'proline']
# dataset = pd.read_csv('./wine.csv', names=['variety', 'alco', 'malic', 'ash', 'alcal', 'mag', 'total',
#                                                'flav', 'nonflav', 'proan', 'color', 'hue', 'OD', 'proline'], skiprows=0)

# # load ionos dataset from csv
# dataset = pd.read_csv('./ionosphere.csv', names=['c', 'd', 'e', 'f', 'g',
#                                                'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
#                                                      'u', 'v', 'w', 'x', 'y', 'z', 'ab', 'bc', 'cd', 'de', 'ef',
#                                                       'fg', 'gh', 'hi', 'variety'], skiprows=0)

# # load vertebral dataset from csv
# names=['a', 'b', 'c', 'd', 'e', 'f', 'variety']
# dataset = pd.read_csv('./vertebral.csv', names=['a', 'b', 'c', 'd', 'e', 'f', 'variety'], skiprows=0)



dataVariety = dataset['variety'].unique()


# randomize the iris data
randomData = dataset.sample(frac=1)
datasize = randomData.shape[0]

KFold = 4
success = 0
confusion_mat = np.zeros([len(dataVariety), len(dataVariety)])
g_unsort = []
testIdX = []
for i in range(KFold):
    # split the train data and test data
    trainData = randomData[:int(datasize * i / KFold) + int(datasize * (i + 1) / KFold):]
    testData = randomData[int(datasize * i / KFold):int(datasize * (i + 1) / KFold)]

    mu = np.zeros((len(dataVariety), trainData.shape[1] - 1, 1))
    cov = np.zeros((len(dataVariety), trainData.shape[1] - 1, trainData.shape[1] - 1))

    prob = []

    # train data
    for i in range(len(dataVariety)):
        classData = trainData[trainData['variety'] == dataVariety[i]]
        mu[i], cov[i] = maxLikelihood(classData)
        prob.append(classData.shape[0] / trainData.shape[0])

    # test data
    for index, row in testData.iterrows():
        g = np.zeros([len(dataVariety)])
        for i in range(len(dataVariety)):
            g[i] = gi(row, mu[i], cov[i], prob[i])
        g_unsort.append(g[0] - g[1])
        testIdX.append(index)
        predict = np.argmax(g)


        if (dataVariety[predict] == row["variety"]):
            success += 1

        real = -1
        for i in range(len(dataVariety)):
            if dataVariety[i] == row["variety"]:
                real = i
        confusion_mat[predict][real] += 1


print("accuracy: %f" %(success / dataset.shape[0]))

# draw confusion matrix
fig, axis = plt.subplots(figsize=(5, 5))
axis.matshow(confusion_mat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confusion_mat.shape[0]):
    for j in range(confusion_mat.shape[1]):
        axis.text(x=j, y=i, s=confusion_mat[i,j], va='center', ha='center')

tick_marks = np.arange(len(dataVariety))
plt.xticks(tick_marks, dataVariety, rotation=0)
plt.yticks(tick_marks, dataVariety)

plt.xlabel('predicted label')
plt.ylabel('true label')
plt.title('Confusion matrix')
plt.show()



if (len(dataVariety) == 2):
    # draw ROC curve
    g_sort = sorted(g_unsort, reverse=True)
    TPR = []
    FPR = []
    labels = np.zeros([dataset.shape[0]])
    predPosProb = np.zeros([dataset.shape[0]])


    for i in range(len(g_sort)):
        TP = 0
        FP = 0
        FN = 0
        TN = 0

        threshold = g_sort[i]
        for j in range(len(g_unsort)):
            if g_unsort[j] >= threshold and dataset.ix[testIdX[j], "variety"] == dataVariety[0]:
                TP += 1
                predPosProb[testIdX[j]] += 1
            elif g_unsort[j] >= threshold and dataset.ix[testIdX[j], "variety"] == dataVariety[1]:
                FP += 1
                predPosProb[testIdX[j]] += 1
            elif g_unsort[j] < threshold and dataset.ix[testIdX[j], "variety"] == dataVariety[0]:
                FN += 1
            elif g_unsort[j] < threshold and dataset.ix[testIdX[j], "variety"] == dataVariety[1]:
                TN += 1
        TPR.append(TP / (TP + FN))
        FPR.append(FP / (FP + TN))


    plt.xlabel('FA')
    plt.ylabel('PD')
    plt.plot(FPR, TPR)
    plt.show()

    # calculate AUC
    auc = 0
    for i in range(len(TPR) - 1):
        auc += (TPR[i] + TPR[i + 1]) * (FPR[i + 1] - FPR[i]) / 2
    print("AUC: %f" %(auc))


