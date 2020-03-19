import numpy as np
import random
from sklearn import metrics

import Header

def HCM(randomData, dataVariety, datasize, c, noise):
    # dataframe to numpy array
    dataLabel = np.array(randomData['variety'])
    dataLabel = dataLabel.reshape([datasize])

    dropData = randomData.drop("variety", axis=1)
    dataArr = np.array(dropData)
    dataList = dataArr.tolist()

    # randomly choose means
    means = random.sample(dataList, c)

    lastCluster = np.zeros([datasize])
    iteration = 0
    while 1:
        iteration += 1

        cluster = np.zeros([datasize])
        for i in range(datasize):
            dataDist = np.zeros([c])
            for j in range(c):
                dataDist[j] = Header.calEucDist(dataArr[i], means[j])
            cluster[i] = np.argmin(dataDist)
        if (lastCluster == cluster).all():
            break

        # recalculate means
        for i in range(c):
            index = np.argwhere(cluster == i)
            clusterData = np.zeros([index.shape[0], dataArr.shape[1]])
            for j in range(index.shape[0]):
                clusterData[j] = dataArr[index[j]]
            means[i] = np.mean(clusterData, axis=0)
        lastCluster = cluster
    print("HCM")
    print("iteration: %d" % iteration)

    predData, predLabel = Header.clustSorting(dataArr, lastCluster)
    if noise == 0:
        trueData, trueLabel = Header.clustSorting(dataArr, dataLabel)
    else:
        trueData, trueLabel, noiseArr = Header.realDataSorting(dataArr, dataLabel, dataVariety, noise)

    Header.plotFig(predData, "HCM_figure1")
    if noise == 0:
        Header.plotFig(trueData, "HCM_figure2")
    else:
        Header.plotNoiseFig(trueData, "HCM_figure2", noiseArr)

    ARI = metrics.adjusted_rand_score(predLabel, trueLabel)
    print("ARI: %g" % ARI)