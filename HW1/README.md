## Methods I have implemented
### I. Bayesian classifier (use Gaussian pdfs with maximum-likelihood estimation)

先將原始dataset的順序隨機排序，然後利用cross validation將dataset分出training data和testing data後，將training data中每個類別的mean和covariance算出來，得到每個類別的maximum likelihood。接著在testing data的部分，一筆⼀筆資料處理，將該筆testing data每個類別的Gaussian pdf算出來來，和該類別的機率(priori probability)相乘後得到Discriminant function，將每個類別的Discriminant function比⼤小，最大值的類別即為該testing data的預測類別。

<img src="https://i.imgur.com/kbWN5dW.png" width="70%">

### II. Naïve-Bayes classifier

先將原始dataset的順序隨機排序，然後利用cross validation將dataset分出training data和testing data後，將training data中每個類別的每個attribute的mean和variance算出來。接著在testing data的部分，⼀筆一筆資料處理，將該筆testing data每個類別的每個attribute的 pdf算出來後相乘，再和該類別的機率(priori probability)相乘後得到posteriori probability，將每個類別的posteriori probability比⼤小，最⼤值的類別即為該testing data的預測類別。

<img src="https://i.imgur.com/WJaPXSL.png" width="80%">

### III. Linear classifier (by Perceptron Algorithm)

先將原始dataset的順序隨機排序，然後利用cross validation將dataset分出training data和testing data後，為了找出decision boundary，設⼀個向量w依序和每筆training data所形成的向量x相乘，如果結果大於0就分到第一類，⼩於等於0就分到第二類。如果預測出來的類別和實際的類別y不同，就更新w為w = w + y * x，之後就如此反覆用w繼續對之後的training data預測分類，由於計算量過於龐大，因此我設定分類達到10次後即停止，⽽效果和分類100次的效果差不多，估計是因為w更新10次後就漸漸收斂，之後的更新也不會變動太大。接著在testing data的部分，⼀筆⼀筆資料處理，將training data時最後得到的w和testing data所形成的向量x相乘，如果結果大於0就分到第一類，⼩於等於0就分到第二類。

<img src="https://i.imgur.com/QwchvIi.png" width="60%">

### IV. Confusion matrix

根據testing data被classifier預測的類別和實際類別，將confusion matrix算出來，再用pyplot將confusion matrix陣列顯⽰出來。

### V. ROC curve

將預測testing data過程中的所有threshold記錄下來來，並將threshold由⼤排到小，依序⽤這些threshold再度將testing data再分類⼀次，如果該testing data的g值⼤於等於threshold就分到第類，否則就分到第⼆類。根據testing data被預測的類別和實際類別，將結果分成TN, FP, FN, TP四種，並算出PD和FA，最後將所有的PD和FA⽤pyplot畫出曲線圖。

### VI. AUC

由於AUC是ROC curve線下的⾯積，所以我將每個ROC curve上前後的點，以計算梯形的⾯積的⽅式來估計兩點間的線下面積，最後再加總起來，得到AUC的估計值。

## Experiments I have done, and the results

### I. Bayesian classifier (use Gaussian pdfs with maximum-likelihood estimation)

經過多次的測試，發現Bayesian classifier在k = 4下的交叉驗證表現最穩，準確率最高，因此以下測試結果皆以k = 4做測試。此外，由於Ionosphere dataset的前兩個feature會使準確率較低，因此將前兩個feature拿掉。

<img src="https://i.imgur.com/vR2Lyds.png" width="80%">
<img src="https://i.imgur.com/EGiFdCh.png" width="80%">

### II. Naïve-Bayes classifier

經過多次的測試，發現Naïve-Bayes classifier在k = 10下的交叉驗證表現最穩，準確率最高，因此以下測試結果皆以k = 10做測試。此外，由於Ionosphere dataset的前兩個feature會使準確率較低，因此將前兩個feature拿掉。

<img src="https://i.imgur.com/k1cpSmB.png" width="80%">
<img src="https://i.imgur.com/cgZNQd8.png" width="80%">
<img src="https://i.imgur.com/yv9VJmn.png" width="80%">

### III. Linear classifier (by Perceptron Algorithm)

經過多次的測試，發現Linear classifier在k = 10下的交叉驗證表現最穩，準確率最高，因此以下測試結果皆以k = 10做測試。此外，由於Ionosphere dataset的前兩個feature會使準確率較低，因此將前兩個feature拿掉。為了讓執行時間不要太久，經過測試發現w值更新10次，就會有不錯的準確率，因此設定w值最多只能更新10次，否則如果dataset並非seperable，程式將無止盡跑下去。

<img src="https://i.imgur.com/rLyG2tI.png" width="80%">
<img src="https://i.imgur.com/nWE3s4N.png" width="80%">

## Analysis - Are the results what I expect? Why?

比較這三個分類器，我發現準確率從高到低依序為Bayesian classifier, Naïve-Bayes classifier, Linear classifier，其中最⾼的是Bayesian classifier，其準確率有時甚至還達到100%，⽽最低的是Linear classifier，對於這樣的結果其實我並不意外。有些dataset不見得能找出⼀條線來完美區分兩類data，總是會有⼀些data和別類的data混合在一起，⽽linear classifier對於分類這種data會不好區分，造成準確率較低，但整體而⾔準確率也算是⾼。  

另外，我發現在不同分類器下不同dataset的準確率⾼低排序相當⼀致，依序為Wine, Iris, Ionosphere, Vertebral，而我認為原因可能跟dataset本身的attribute分佈有關。如果dataset 中不同類別的attribute分佈差很多的話，則dataset就會容易被分類。反之，如果dataset中不同類別的attribute分佈差太少的話，則dataset在分類時就容易被誤判。
