import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import random
import cv2

# load iris dataset from csv
def loadIris():
    names = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width', 'variety']
    Dataset = pd.read_csv('./iris.csv', names=['sepal.length', 'sepal.width', 'petal.length', 'petal.width', 'variety'], skiprows=0)
    DataVariety = Dataset['variety'].unique()
    return Dataset, DataVariety, names

# load wine dataset from csv
def loadWine():
    names=['variety', 'alco', 'malic', 'ash', 'alcal', 'mag', 'total', 'flav', 'nonflav', 'proan', 'color', 'hue', 'OD', 'proline']
    Dataset = pd.read_csv('./wine.csv', names=['variety', 'alco', 'malic', 'ash', 'alcal', 'mag', 'total',
                                                   'flav', 'nonflav', 'proan', 'color', 'hue', 'OD', 'proline'], skiprows=0)
    DataVariety = Dataset['variety'].unique()
    return Dataset, DataVariety, names


# load ionos dataset from csv
def loadIonos():
    names = ['c', 'd', 'e', 'f', 'g',
             'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
             'u', 'v', 'w', 'x', 'y', 'z', 'ab', 'bc', 'cd', 'de', 'ef',
             'fg', 'gh', 'hi', 'variety']
    Dataset = pd.read_csv('./ionosphere.csv', names=['c', 'd', 'e', 'f', 'g',
                                                     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                                                     'u', 'v', 'w', 'x', 'y', 'z', 'ab', 'bc', 'cd', 'de', 'ef',
                                                     'fg', 'gh', 'hi', 'variety'], skiprows=0)
    DataVariety = Dataset['variety'].unique()
    return Dataset, DataVariety, names


# load vertebral dataset from csv
def loadVertebral():
    names = ['a', 'b', 'c', 'd', 'e', 'f', 'variety']
    Dataset = pd.read_csv('./vertebral.csv', names=['a', 'b', 'c', 'd', 'e', 'f', 'variety'], skiprows=0)
    DataVariety = Dataset['variety'].unique()
    return Dataset, DataVariety, names

# load face dataset
def loadFaceImg():
    img1 = cv2.imread('mP1.bmp', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('fP1.bmp', cv2.IMREAD_GRAYSCALE)

    img1Data = np.zeros([100, 1600])
    img2Data = np.zeros([100, 1600])
    originX = 0
    originY = 0
    img1Idx = 0
    img2Idx = 0
    idx = 0
    for p in range(10):
        for q in range(10):
            for i in range(40):
                for j in range(40):
                    img1Data[img1Idx][idx] = img1[originX + i][originY + j]
                    img2Data[img1Idx][idx] = img2[originX + i][originY + j]
                    idx += 1
            img1Idx += 1
            img2Idx += 1
            idx = 0
            originY += 40
        originY = 0
        originX += 40

    img1Flip = cv2.flip(img1Data, 1)
    img2Flip = cv2.flip(img2Data, 1)

    img1DF = pd.DataFrame(img1Data)
    img1DF['variety'] = 'male'
    img2DF = pd.DataFrame(img2Data)
    img2DF['variety'] = 'female'

    img1FDF = pd.DataFrame(img1Flip)
    img1FDF['variety'] = 'male'
    img2FDF = pd.DataFrame(img2Flip)
    img2FDF['variety'] = 'female'

    frames = [img1DF, img2DF, img1FDF, img2FDF]
    Dataset = pd.concat(frames, ignore_index=True)
    DataVariety = Dataset['variety'].unique()

    # randomize data
    randomData = Dataset.sample(frac=1)
    datasize = randomData.shape[0]

    # split the train data and test data
    trainData = randomData[:int(datasize * 0.75)]
    testData = randomData[int(datasize * 0.75):]

    return trainData, testData, DataVariety

def loadFaceImg2():
    img = cv2.imread('facesP1.bmp', cv2.IMREAD_GRAYSCALE)

    imgTrain = np.zeros([64, 1600])
    imgTest = np.zeros([16, 1600])

    originX = 0
    originY = 0
    imgTrainIdx = 0
    imgTestIdx = 0
    idx = 0
    for p in range(5):
        for q in range(16):
            for i in range(40):
                for j in range(40):
                    if p != 4:
                        imgTrain[imgTrainIdx][idx] = img[originX + i][originY + j]
                    else:
                        imgTest[imgTestIdx][idx] = img[originX + i][originY + j]
                    idx += 1
            if p != 4:
                imgTrainIdx += 1
            else:
                imgTestIdx += 1
            idx = 0
            originY += 40
        originY = 0
        originX += 40

    imgTrainFlip = cv2.flip(imgTrain, 1)
    imgTrainDF = pd.DataFrame(imgTrain)
    imgTrainFlipDF = pd.DataFrame(imgTrainFlip)
    trainFrame = pd.concat([imgTrainDF, imgTrainFlipDF], axis=0, ignore_index=True)

    imgTestFlip = cv2.flip(imgTest, 1)
    imgTestDF = pd.DataFrame(imgTest)
    imgTestFlipDF = pd.DataFrame(imgTestFlip)
    testFrame = pd.concat([imgTestDF, imgTestFlipDF], axis=0, ignore_index=True)

    trainVarArr = np.zeros([128, 1])
    var = 0
    for j in range(128):
        trainVarArr[j] = var
        var += 1
        if var % 16 == 0:
            var = 0

    testVarArr = np.zeros([32, 1])
    var = 0
    for j in range(32):
        testVarArr[j] = var
        var += 1
        if var % 16 == 0:
            var = 0

    trainVarDF = pd.DataFrame(trainVarArr, columns=['variety'])
    trainDataset = pd.concat([trainFrame, trainVarDF], axis=1)

    testVarDF = pd.DataFrame(testVarArr, columns=['variety'])
    testDataset = pd.concat([testFrame, testVarDF], axis=1)

    DataVariety = trainDataset['variety'].unique()

    return trainDataset, testDataset, DataVariety


# calculate accuracy
def calAccuracy(Success, Dataset):
    accuracy = Success / Dataset.shape[0]
    print("accuracy: %f" %(accuracy))

# draw confusion matrix
def drawConfusionMat(Confusion_mat, DataVariety):
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.matshow(Confusion_mat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(Confusion_mat.shape[0]):
        for j in range(Confusion_mat.shape[1]):
            axis.text(x=j, y=i, s=Confusion_mat[i,j], va='center', ha='center')

    tick_marks = np.arange(len(DataVariety))
    plt.xticks(tick_marks, DataVariety, rotation=0)
    plt.yticks(tick_marks, DataVariety)

    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.title('Confusion matrix')
    plt.show()

# draw ROC curve & calculate AUC
def ROC_AUC(Dataset, DataVariety, G, TestIdx):
    if (len(DataVariety) == 2):
        # draw ROC curve
        g_sort = sorted(G, reverse=True)
        TPR = []
        FPR = []
        labels = np.zeros([Dataset.shape[0]])
        predPosProb = np.zeros([Dataset.shape[0]])

        for i in range(len(g_sort)):
            TP = 0
            FP = 0
            FN = 0
            TN = 0
            threshold = g_sort[i]
            for j in range(len(G)):
                if G[j] >= g_sort[i] and Dataset.ix[TestIdx[j], "variety"] == DataVariety[0]:
                    TP += 1
                    predPosProb[TestIdx[j]] += 1
                    labels[TestIdx[j]] = 1
                elif G[j] >= g_sort[i] and Dataset.ix[TestIdx[j], "variety"] == DataVariety[1]:
                    FP += 1
                    predPosProb[TestIdx[j]] += 1
                    labels[TestIdx[j]] = 0
                elif G[j] < g_sort[i] and Dataset.ix[TestIdx[j], "variety"] == DataVariety[0]:
                    FN += 1
                    labels[TestIdx[j]] = 1
                else:
                    TN += 1
                    labels[TestIdx[j]] = 0
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
        print("AUC: %f" % (auc))
