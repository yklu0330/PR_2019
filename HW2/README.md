## Task
### I. FLD
Use only two-class datasets (at least three, including the gender classification data) for this task. Apply FLD to find the projection direction that preserves the most class separability. Compute the separability measures before and after the projection. Also plot the resulting ROC curves as well as AUC. For datasets also used in the first assignment, be sure to compare with the results there from the linear classifier implemented then.

### II. PCA and classification
Take the datasets used in the first assignment, apply PCA to them, and then repeat the experiments there. The objective is for you to see how PCA affects classification performance. Pay attention to how you choose the retained the dimensions as this is an important factor here. You do not need to repeat every experiments you did in the first assignment, but you should do enough for you to make meaningful observations about the aforementioned objective. Also note whether the effects of PCA are different for different classifier types. Here you need to split your data into training and validation subsets, and the projection matrices are only derived from the training subset. The same splitting should be used when comparing different classifiers or settings.

### III. Eigenface and classification
Here your goal is to use the provided face image datasets for gender classification (fP1.bmp and mP1.bmp) and face recognition (facesP1.bmp, 5 images for each class), respectively, after PCA (the "eigenface" version) is applied to the images. As mentioned in the class, it is a common practice to generate horizontally flipped version of the face images (training subset only).

## Methods I have implemented
### I. FLD

先將原始dataset的順序隨機排序，然後將前3/4筆data當作training data，⽽剩下1/4的data當作test data。接著將training data的scatter matrix Sw和Sb依照公式算出來，再利用Sw和 training data的mean將w算出來。在test data的部分，⼀筆⼀筆資料處理，將每筆test data用w做投影，做法是算出wTx，得到投影後的向量。最後，再將投影後的向量用上次作業的Bayesian classifier, Naive-Bayes classifier, linear classifier預測每筆test data的類別。此 外，在投影前後都有依據公式算出separability measures，以便分析投影的效果。

<img src="https://i.imgur.com/8PFbnhb.png" width="70%">

### II. PCA and classification

先將原始dataset的順序隨機排序，然後將前3/4筆data當作training data，⽽剩下1/4的data當作test data。接著分別將training data和test data做標準化，再⽤標準化後的training data算出covariance matrix，再求出covariance的eigenvalue和eigenvector，取出最大的L個 eigenvalue的eigenvector當作投影矩陣A。此處的L值是降維後的維度，經過實測，隨著dataset的不同，每種dataset有各⾃適合的L值，⽽且L值也會受到分類器的影響。最後，利用A矩陣將training data和test data做投影，並將投影後的資料⽤上次作業的Bayesian classifier, Naive-Bayes classifier, linear classifier來預測每筆test data的類別。

<img src="https://i.imgur.com/mj5H5Ql.png" width="20%">


### III. Eigenface and classification

作業的eigenface實作部分，是使⽤性別辨識和⼈臉辨識兩種dataset。由於兩種dataset結構
的不同，因此⽤了不同的資料取樣方法。

性別辨識的部分，讀取每張臉的影像中每個pixel的灰階值，再將每張臉的影像⽔平翻轉，得到翻轉後的臉的影像，如此⼀來得到原本影像中2倍數量的臉。之後，再將這些臉的dataset順序隨機排序，然後將前3/4筆data當作training data，⽽剩下1/4的data當作test data。

⼈臉辨識的部分，由於每種臉的原始dataset只有5個，為了避免隨機排序後取得的training data中，某些種類的臉取得樣本太少(例如1, 2個甚⾄沒有)，導致無法準確分類，因此我不採取隨機取樣。我將每種臉的前4張影像以及這4張臉水平翻轉後的影像分到training data，每種臉剩下的1張影像分到test data。

將dataset分成training data和test data後，先將training data標準化，把每筆training data當作向量放到X矩陣的每⼀行。接著算出 ![](http://latex.codecogs.com/gif.latex?X^TX) ，並得到 ![](http://latex.codecogs.com/gif.latex?X^TX) 的eigenvalue和eigenvector， 再將X乘以eigenvector得到新的eigenvector並normalize。之後，取出最⼤的L個eigenvalue的新的eigenvector當作投影矩陣A。此處的L值是降維後的維度，經過實測，隨著dataset的不同，每種dataset有各⾃適合的L值，而且L值也會受到分類器的影響。最後，利用A矩陣將training data和test data做投影，並將投影後的資料用上次作業的Bayesian classifier, Naive-Bayes classifier, linear classifier來預測每筆test data的類別。

## Experiments I have done, and the results

### I. FLD

經過多次的測試，發現FLD在測試的每個dataset表現不錯，其ROC curve和auc值都比上次作業實作的linear classifier略高，⽽且透過計算出來的separability measures會發現，投影後資料確實有變得比較集中。然⽽，FLD在gender dataset，有時auc值會特別低，⼤約0.2，原因⾄今我仍想不透。



<img src="https://i.imgur.com/nTEWxHg.png" width="90%">
<img src="https://i.imgur.com/XCdN24I.png" width="60%">

### II. PCA and classification

經過多次的測試，發現如果dataset在10維以下，降維到3維效果最好;如果dataset在10維以上、20維以下，降維到10維效果最好;如果dataset在20維以上，降維到20維效果最好。我覺得可能是因為如果維度太低，可能會因為資訊太少而不準確;如果維度太高，可能會因為overfitting⽽準確率降低。除此之外，和上次作業的實作結果相比，不論使⽤哪個dataset，⽤PCA降維後的分類表現有時候比較差，我覺得原因應該是因為降維後有些資訊損失，包含的種類資訊太少，導致誤判的可能性較高。

<img src="https://i.imgur.com/LydWY4v.png" width="70%">
<img src="https://i.imgur.com/3z58q5o.png" width="69%">
<img src="https://i.imgur.com/HoB3V0b.png" width="75%">
<img src="https://i.imgur.com/70cRXVf.png" width="69%">
<img src="https://i.imgur.com/mQ3ghqS.png" width="70%">


### III. Eigenface and classification

性別辨識的部分，我採⽤了Bayes classifier, Naïve-Bayes classifier, Linear classifer做最後的分類。⼈臉辨識的部分，經過多次測試，由於Bayes classifier不知道為什麼，在算probability density function的時候會⼀直算出singular matrix導致程式無法繼續，所以最後我只用Naïve-Bayes classifier分類。 

為了瞭解降維對於資料分類表現的影響，在性別辨識的時候，我將資料維度從2到200的準確率和auc值做成折線圖，結果發現準確率和auc值在維度2~30的時候逐漸攀升，但到了30以後卻逐漸下降。⽽在⼈臉辨識的時候，到了維度100以後準確率和auc值也逐漸下降。我覺得可能是維度太高的話會overfitting，導致分類表現反⽽會變得更差。

<img src="https://i.imgur.com/H2YrgVS.png" width="80%">
<img src="https://i.imgur.com/sSyFQZB.png" width="85%">
<img src="https://i.imgur.com/L8VJxkH.png" width="82%">

<img src="https://i.imgur.com/FtBeZzT.png" width="50%">
<img src="https://i.imgur.com/GvnnHiN.png" width="50%">

## Analysis - Are the results what I expect? Why?

比較這三種⽅法，我發現PCA表現得最好，但PCA和eigenface都有個很大的缺點，就是受維度影響太⼤大。如果沒有取適當數量的維度，分類表現就會很差，因此選擇適當的維度很重要。FDA的缺點就是只能分類兩種class的dataset，但FDA整體表現不差。和上次的作業相比，在⽤相同分類器的情況下，降維後的表現明顯差⼀些，甚至表現很不穩定，準確率有時候很高，有時候很低，看來可能是因為降維後training data的資訊變少，這時training data取樣的好壞就會影響很大。整體⽽言，我認為這次作業的結果比我想像的還差，雖然原本就有預料到降維後可能表現較差，但沒想到會差這麼多，尤其是⼈臉辨識的表現並不是很好。另外，我發現在上次作業中，不同分類器下不同dataset的準確率高低排序相當一致，依序為Wine, Iris, Ionosphere, Vertebral，而這次作業在降維後，不同dataset的準確率高低排序仍和上次作業的⼀致，看來降維前後不會影響dataset本⾝好壞的排序。


