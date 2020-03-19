import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import statistics as stat
import random
from scipy.stats import norm
import math

def pdf(x, mean, var):
    pdfVal = 1 / math.sqrt(2 * math.pi * var) * math.exp(-pow((x - mean), 2) / (2 * var))
    return pdfVal

# load iris dataset from csv
names=['sepal.length', 'sepal.width', 'petal.length', 'petal.width', 'variety']
dataset = pd.read_csv('./iris.csv', names=['sepal.length', 'sepal.width', 'petal.length', 'petal.width', 'variety'], skiprows=0)


# # load wine dataset from csv
# names=['variety', 'alco', 'malic', 'ash', 'alcal', 'mag', 'total', 'flav', 'nonflav', 'proan', 'color', 'hue', 'OD', 'proline']
# dataset = pd.read_csv('./wine.csv', names=['variety', 'alco', 'malic', 'ash', 'alcal', 'mag', 'total',
#                                                'flav', 'nonflav', 'proan', 'color', 'hue', 'OD', 'proline'], skiprows=0)


# # load ionos dataset from csv
# names=['c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
#                                                      'u', 'v', 'w', 'x', 'y', 'z', 'ab', 'bc', 'cd', 'de', 'ef',
#                                                      'fg', 'gh', 'hi', 'variety']
# dataset = pd.read_csv('./ionosphere.csv', names=['c', 'd', 'e', 'f', 'g',
#                                                'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
#                                                      'u', 'v', 'w', 'x', 'y', 'z', 'ab', 'bc', 'cd', 'de', 'ef',
#                                                      'fg', 'gh', 'hi', 'variety'], skiprows=0)


# # load vertebral dataset from csv
# names=['a', 'b', 'c', 'd', 'e', 'f', 'variety']
# dataset = pd.read_csv('./vertebral.csv', names=['a', 'b', 'c', 'd', 'e', 'f', 'variety'], skiprows=0)

dataVariety = dataset['variety'].unique()

# randomize the input data
randomData = dataset.sample(frac=1)
datasize = randomData.shape[0]

KFold = 10
success = 0
confusion_mat = np.zeros([len(dataVariety), len(dataVariety)])
g = []
testIdX = []
for i in range(KFold):
    # split the train data and test data
    trainData = randomData[:int(datasize * i / KFold) + int(datasize * (i + 1) / KFold):]
    testData = randomData[int(datasize * i / KFold):int(datasize * (i + 1) / KFold)]

    # train data
    mean = np.zeros((len(dataVariety), trainData.shape[1] - 1))
    var = np.zeros((len(dataVariety), trainData.shape[1] - 1))

    prob = []

    for i in range(len(dataVariety)):
        classData = trainData[trainData['variety'] == dataVariety[i]]
        classData = classData.drop("variety", axis=1)
        prob.append(classData.shape[0] / trainData.shape[0])
        for j in range(classData.shape[1]):
            classDataAttr = classData.iloc[:, j]
            mean[i][j] = stat.mean(classDataAttr)
            var[i][j] = stat.variance(classDataAttr)

    # test data
    for index, row in testData.iterrows():
        pdfVal = np.zeros((len(dataVariety), testData.shape[1] - 1))
        post = np.zeros([len(dataVariety)])
        for i in range(len(dataVariety)):
            multiPdf = 1
            newj = 0
            for j in range(testData.shape[1]):
                if names[j] == "variety":
                    continue
                pdfVal[i][newj] = pdf(row.ix[j], mean[i][newj], var[i][newj])
                multiPdf = multiPdf * pdfVal[i][newj]
                newj = newj + 1
            post[i] = prob[i] * multiPdf
        g.append(post[0] - post[1])
        testIdX.append(index)
        predict = np.argmax(post)

        if (dataVariety[predict] == row["variety"]):
            success += 1

        real = -1
        for i in range(len(dataVariety)):
            if dataVariety[i] == row["variety"]:
                real = i
        confusion_mat[predict][real] += 1

accuracy = success / dataset.shape[0]
print("accuracy: %f" %(accuracy))
# print(confusion_mat)


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
    g_sort = sorted(g, reverse=True)
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
        for j in range(len(g)):
            if g[j] >= threshold and dataset.ix[testIdX[j], "variety"] == dataVariety[0]:
                TP += 1
                predPosProb[testIdX[j]] += 1
                labels[testIdX[j]] = 1
            elif g[j] >= threshold and dataset.ix[testIdX[j], "variety"] == dataVariety[1]:
                FP += 1
                predPosProb[testIdX[j]] += 1
                labels[testIdX[j]] = 0
            elif g[j] < threshold and dataset.ix[testIdX[j], "variety"] == dataVariety[0]:
                FN += 1
                labels[testIdX[j]] = 1
            elif g[j] < threshold and dataset.ix[testIdX[j], "variety"] == dataVariety[1]:
                TN += 1
                labels[testIdX[j]] = 0
        TPR.append(TP / (TP + FN))
        FPR.append(FP / (FP + TN))


    plt.xlabel('FA')
    plt.ylabel('PD')
    plt.plot(FPR, TPR)
    plt.show()

    auc = 0
    for i in range(len(TPR) - 1):
        auc += (TPR[i] + TPR[i + 1]) * (FPR[i + 1] - FPR[i]) / 2
    print("AUC: %f" %(auc))

