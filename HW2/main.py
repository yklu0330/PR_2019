import pandas as pd

import Header
import Likelihood
import Bayes
import LinearClassifier
import FLD
import PCA
import Eigenface

# dataset, dataVariety, names = Header.loadIris()
# dataset, dataVariety, names = Header.loadWine()
# dataset, dataVariety, names = Header.loadIonos()
# dataset, dataVariety, names = Header.loadVertebral()
trainData, testData, dataVariety = Header.loadFaceImg()
# trainData, testData, dataVariety = Header.loadFaceImg2()

# g, testIdx, success, confusion_mat = Likelihood.likelihood(dataset, dataVariety)
# g, testIdx, success, confusion_mat = Bayes.bayes(dataset, dataVariety, names)
# g, testIdx, success, confusion_mat = LinearClassifier.linearClassifier(dataset, dataVariety)

# frames = [trainData, testData]
# dataset = pd.concat(frames, ignore_index=True)

# g, testIdx = FLD.FLD(dataset, dataVariety)
# Header.ROC_AUC(dataset, dataVariety, g, testIdx)

# PCA.PCA(dataset, dataVariety, names)
Eigenface.eigenface(trainData, testData, dataVariety)

# Header.calAccuracy(success, dataset)
# Header.drawConfusionMat(confusion_mat, dataVariety)