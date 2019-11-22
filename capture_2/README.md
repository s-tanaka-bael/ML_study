# Pythonで始める機械学習

常に書くっぽいやつはめもっておく

```
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import IPython
import sklearn
import mglearn
%matplotlib inline
```


## 2章：教師あり学習


### pythonぽい書き方のやつ
```
print("Sample counts per class : \n {}".format(
    {n: v for n , v in zip(cancer.target_names , np.bincount(cancer.target))})
)
>> Sample counts per class : 
>> {'malignant': 212, 'benign': 357}
```

#### np.bincount
```
array = [0,1,1,5]
count = np.bincount(array)  -> [1,2,0,0,0,1] 
ans = np.argmax(count)  -> 1
```
bincountの引数の配列の要素内の値が何個あるかを調べてくれる。
cancer例題の場合はtarget_nameの順に0か1（陽性か陰性か）が入っている。


### k-最近傍法の特徴
(以下の部分、ほんとか怪しいと思っている)

!["knn"](img/k-neighbors.png)

左から近傍点1,3,9

この境界線を決定境界という


* 近傍点の数を1とかにすると、学習データに対して最適になる
    * = 複雑度の高いモデルに対応する
* 近傍点の数を増やすと滑らかになる
    * 滑らかな境界はより単純なモデルに最適化する
* 極端なケースとして、近傍点を全てのデータポイントと同じにすると、そのデータセットに対して最適なモデルになる
    * これが過学習？

#### いい感じの決定境界探し

```
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train , X_test , y_train , y_test = train_test_split(
    cancer.data , cancer.target , stratify = cancer.target , random_state = 66)

training_accuracy = []
test_accuracy = []

neighbors_settings = range(1,11)

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train,y_train)
    training_accuracy.append(clf.score(X_train,y_train))
    test_accuracy.append(clf.score(X_test,y_test))
    
plt.plot(neighbors_settings , training_accuracy , label="training accuracy")
plt.plot(neighbors_settings , test_accuracy , label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
```

!["knn2"](img/k-neighbors2.png)

近傍点が6のあたりが、テストのデータを試した感じいい感じ。


### k-近傍回帰

k-最近傍法の亜種

KNeighborsRegressorで使える

```
fig , axes = plt.subplots(1,3,figsize=(15,4))
line = np.linspace(-3,3,1000).reshape(-1,1)
for n_neighbors , ax in zip([1,3,9], axes):
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train,y_train)
    ax.plot(line , reg.predict(line))
    ax.plot(X_train,y_train, '^' , c=mglearn.cm2(0) , markersize=8)
    ax.plot(X_test,y_test, 'v' , c=mglearn.cm2(1) , markersize=8)
    
    ax.set_title(
        "{} neighbors train score : {:.2f} test score : {:.2f}".format(
            n_neighbors , reg.score(X_train , y_train),
            reg.score(X_test,y_test)))
    
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")

axes[0].legend(["Model predictions" , "Training data/target","Test data/target"] , loc="best")
```

!["knn3"](img/k-neighbors3.png)

3あたりが、テストデータにフィットしていそう

### 最近傍法、近傍回帰の注意

* データポイントや特徴量の多いデータには向かない（遅い）ので、実際にはあまり使われない
* けど、単純にデフォでやっただけでも精度は結構出るので、まずはここを起点に考えると良い


### 線形回帰

#### 最小二乗法


```
from sklearn.linear_model import LinearRegression
X,y = mglearn.datasets.make_wave(n_samples=60)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)

lr = LinearRegression().fit(X_train,y_train)

display(lr.coef_)
display(lr.intercept_)

>> array([0.39390555])
>> -0.031804343026759746
```

線形回帰に調整パラメーターはない。

* 傾きを表すパラメーター(w)は、重み、もしくは係数(coefficient)といわれ coef_ に格納される
* オフセットもしくは切片(intercept , b)はintercept_ に格納される
* coef_ とか intercept_ の最後のアンダーバーは訓練データから得られたパラメーターを表す
* scikit-learnではユーザーが設定したパラメーターと明確に区分するために、こういう仕様になっている。


#### 線形回帰の注意

* データセットが多くの特徴量を持つ場合（ボストンの例だと104の特徴量で506のサンプル)は過剰適合してしまいがち
* 訓練セットのscoreは高いがテストセットへの適合率が低い時とかはその兆候

#### そんな時に、リッジ回帰

```
from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train,y_train)
display(ridge.score(X_train,y_train))
display(ridge.score(X_test,y_test))

>> 0.8857966585170941
>> 0.7527683481744755
```

* リッジ回帰は alpha 引数で正則化(せいそくか)できる
* デフォルトは alpha = 1 
* ↑で見れるように、リッジ回帰は訓練データへの適合は低いけど、テストデータへの適合は高い（より汎化している）
* リッジ回帰は過剰適合の危険は少ない
* 最良の alpha はデータセットに依存する


```
ridge10 = Ridge(alpha=10).fit(X_train,y_train)
display(ridge10.score(X_train,y_train))
display(ridge10.score(X_test,y_test))

>> 0.7882787115369614
>> 0.6359411489177311
```

* alphaが増えるほと過剰適合していく

```
ridge01 = Ridge(alpha=0.1).fit(X_train,y_train)
display(ridge01.score(X_train,y_train))
display(ridge01.score(X_test,y_test))

>> 0.9282273685001987
>> 0.7722067936479814
``` 

* alphaが減ると適合不足になる。

それをグラフでみるには以下

```
plt.plot(ridge.coef_ , 's' , label="Ridge alpha=1")
plt.plot(ridge10.coef_ , 's' , label="Ridge alpha=10")
plt.plot(ridge01.coef_ , 's' , label="Ridge alpha=0.1")

plt.plot(lr.coef_ , 'o' ,label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0,0,len(lr.coef_))
plt.ylim(-25,25)
plt.legend()
```

!["ridge"](img/ridge.png)

* x軸は coef_ の要素を表す
* x=0 は最初の特徴量に対する係数、x=1は2番目の特徴量に対する係数となって、100まで続いている
* alpha=10 の時は殆どの特徴量が-3から3の範囲に収まっている
* alpha=1 はちょっと上下の範囲がばらついている
* alpha=0.1 の時はかなりばらついている


#### Lasso

Ridgeとは別の線形回帰のやつ
* いくつかのデータを完全に無視するのが特徴

```
from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_train , y_train)
display(lasso.score(X_train,y_train))
display(lasso.score(X_test,y_test))
np.sum(lasso.coef_ != 0)

>> 0.29323768991114596
>> 0.20937503255272272
>> 4
```

* デフォだとめちゃくちゃ弱い
* 4つしか特徴量使ってない

```
lasso001 = Lasso(alpha=0.01 , max_iter=100000).fit(X_train,y_train)

display(lasso001.score(X_train,y_train))
display(lasso001.score(X_test,y_test))
display(np.sum(lasso001.coef_ != 0))

>> 0.8962226511086497
>> 0.7656571174549983
>> 33
```

* alphaが小さくなるとより複雑なデータに適合するようになる

```
lasso00001 = Lasso(alpha=0.0001 , max_iter=100000).fit(X_train,y_train)

display(lasso00001.score(X_train,y_train))
display(lasso00001.score(X_test,y_test))
display(np.sum(lasso00001.coef_ != 0))

>> 0.9507158754515462
>> 0.6437467421273558
>> 96
```
* alpha小さくしすぎると正則化の効果が薄れて過剰適合する


#### クラス分類の為の線形モデル

* ロジスティック回帰と線形サポートベクターマシン

```
# ロジスティック回帰
linear_model.LogisticRegression 

# 線形サポートベクタマシン
svm.LinearSVC
```

forgeデータでおためし

```
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

X,y = mglearn.datasets.make_forge()

fig, axes = plt.subplots(1,2, figsize=(10,3))

for model, ax in zip([LinearSVC(),LogisticRegression()], axes):
    clf = model.fit(X,y)
    mglearn.plots.plot_2d_separator(clf , X , fill=False , eps=0.5 , ax=ax , alpha=.7)
    mglearn.discrete_scatter(X[:,0] , X[:,1] , y , ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")

axes[0].legend
```

!["LRvsSVC"](img/LRvsSVC.png)

* LogisticRegressionとLinearSVCの正則化強度を決定するパラメータはCと言われる
* Cが大きくなると正則化が弱くなる（訓練データに適合度をあげる）
* Cが小さくなるとデータポイントの「大多数」に対して適合しようとする（この辺が面白い特徴らしい）


!["LRvsSVC2"](img/LRvsSVC2.png)

* Cが小さい〜大きいまでの決定境界
* 小さすぎると境界が水平に近くなって、大きすぎると過剰適合


別のデータでやってみる

```
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data , cancer.target , stratify=cancer.target , random_state = 42)
logreg = LogisticRegression().fit(X_train,y_train)

display(logreg.score(X_train,y_train))
display(logreg.score(X_test,y_test))

>> 0.9530516431924883
>> 0.958041958041958
```

#### 線形モデルによる多クラス分類

* 基本線形モデルは多クラス分類できないので、1対nでそれっぽい感じにする

```
from sklearn.datasets import make_blobs

X,y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:,0] , X[:,1] , y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1" , "Class 2"])
```

!["LRvsSVC3"](img/LRvsSVC3.png)

```
linear_svm = LinearSVC().fit(X,y)
display(linear_svm.coef_.shape)
display(linear_svm.intercept_.shape)

>> (3, 2)
>> (3,)
```


```
mglearn.discrete_scatter(X[:,0] , X[:,1] , y)
line = np.linspace(-15,15)
for coef , intercept , color in zip(linear_svm.coef_ , linear_svm.intercept_ , ['b','r','g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)

plt.ylim(-10 , 15)
plt.xlim(-10 , 8)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0","Class 1","Class 2", "Line class 0","Line class 1","Line class 2"], loc=(1.01,0.3))

```

* それぞれのクラスの決定境界はこう

!["LRvsSVC4"](img/LRvsSVC4.png)


```
mglearn.plots.plot_2d_classification(linear_svm , X , fill=True , alpha=.7)
mglearn.discrete_scatter(X[:,0] , X[:,1] , y)
ine = np.linspace(-15,15)
for coef , intercept , color in zip(linear_svm.coef_ , linear_svm.intercept_ , ['b','r','g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)

plt.ylim(-10 , 15)
plt.xlim(-10 , 8)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0","Class 1","Class 2", "Line class 0","Line class 1","Line class 2"], loc=(1.01,0.3))
```

* こんな感じ

!["LRvsSVC5"](img/LRvsSVC5.png)


#### 利点・欠点・パラメータ

* 回帰モデルではalpha
* LinbearSVCとLogisticRegressionではC
* alphaが大きいとき、Cが小さいときは単純なモデルに適合する
* 線形モデルは高速、大きいデータセットにも適用できる
* デメリット何書いてるかよくわからんかった



#### ナイーブベイズクラス分類器

* 線形モデルによく似た分類器
* 高速だけど、LogisticRegressionとかLinearSVCとかより汎化性能が劣る場合が多い
* scikit-learn には GaussianNB , BernoulliNB , MultinomialNB が実装されている
* GaussianNBは任意の連続値データに適用できる
* BernoulliNBは2値データを仮定されている
* MultinomialNBはカウントデータを仮定している
    * カウントデータは例えば、文中に出てくる単語の出現数など


#### 決定木

* Yes/No で答えられる質問で構成された、階層的な木構造を学習する
* 決定木の学習は正解に最も早く辿り着けるようなYes/No型の質問を学習すること
* 個々のテストは1つの特徴量しかもたない(Yes/Noなので)常に軸に平行な境界を持つ
* 1つの対象値になるまで(pureという)やると過剰適合
* ↑葉が純粋、みたいな言い方をする
* それを防ぐために　事前枝刈り(pre-pruning)　、　事後枝刈り(post-pruning)　を行う
* scikit-learnでは、決定木は DecisionTreeRegressor(連続値に使う) と DecisionTreeClassifier(離散値に使う) が実装されている
* どちらも事前枝刈りしか実装されていない

事前枝刈り

```
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(
    cancer.data , cancer.target , stratify=cancer.target , random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train,y_train)
display(tree.score(X_train,y_train))
display(tree.score(X_test,y_test))

>> 1.0
>> 0.9370629370629371
```

* ↑　は訓練セットに対して100%、pureになっている（過剰適合している）
* これを max_depth=4 にしてやると...


```
tree = DecisionTreeClassifier(max_depth=4 , random_state= 0)
tree.fit(X_train,y_train)

display(tree.score(X_train,y_train))
display(tree.score(X_test,y_test))

>> 0.9882629107981221
>> 0.951048951048951
```

!["Tree Dot"](img/tree_dot.png)

こんな感じで出る


#### 特徴量の重要度

* 個々の特徴量がどの程度重要かを示す割合
* 0-1の間で表され、総和は1
* 0がまったく使われていない
* 1は完全にターゲットを予測できる

```
print("Feature Importances:\n{}".format(tree.feature_importances_))

>> Feature Importances:
[0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.01019737 0.04839825
 0.         0.         0.0024156  0.         0.         0.
 0.         0.         0.72682851 0.0458159  0.         0.
 0.0141577  0.         0.018188   0.1221132  0.01188548 0.        ]
 ```

特徴量の重要度の可視化

```
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_ , align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

plot_feature_importances_cancer(tree)
```

!["Feature_Importance"](img/Feature_Importance.png)

* どの特徴量が重要かを表すが、それがどういう結果になるかとは別なので注意



#### DecisionTreeRegressor

* 回帰決定木
* 使い方はクラス分類決定木とほとんど同じ
* 回帰の場合は外挿(extrapolate)ができない。(訓練データのレンジの外側には対応しない)

```
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]

X_train = data_train.date[:,np.newaxis]
y_train = np.log(data_train.price)

tree = DecisionTreeRegressor().fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train,y_train)


X_all = ram_prices.date[:,np.newaxis]

pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)

price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)

plt.semilogy(data_train.date, data_train.price , label="Training data")
plt.semilogy(data_test.date, data_test.price , label="Test data")
plt.semilogy(ram_prices.date, price_tree, label="Tree prediction")
plt.semilogy(ram_prices.date, price_lr , label="Linear prediction")
plt.legend()
```


!["LRvsDTR"](img/LRvsDTR.png)

* 線形回帰の方はテストデータに対してもそこそこフィットしている
* 回帰決定木は訓練データの外側のデータには全く適合していない
* ただし、訓練データの範囲内には「完璧に」適合している
* 決定木は事前枝刈をしたとしても単体では過剰適合しがちで、実際には次に見るアンサンブル法が用いられる


### ランダムフォレスト

* 決定木のデメリットは過剰適合しやすい事
* ランダムフォレストとは、複数の決定木をたくさん集めたもので、個々の決定木は過剰適合していてもそれらを寄せ集めて平均を取ることで過剰適合の度合いを減らす手法


```
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons

X , y = make_moons(n_samples=100 , noise=0.25 , random_state= 3)
X_train , X_test , y_train , y_test = train_test_split(X,y,stratify = y,random_state=42)

forest = RandomForestClassifier(n_estimators=5 , random_state=2)
forest.fit(X_train,y_train)

fig , axes = plt.subplots(2,3,figsize=(20,10))
for i , (ax, tree) in enumerate(zip(axes.ravel(),forest.estimators_)):
    ax.set_title("Tree {}".format(i))
    mglearn.plots.plot_tree_partition(X_train,y_train,tree,ax=ax)

mglearn.plots.plot_2d_separator(forest,X_train,fill=True,ax=axes[-1,-1],alpha=.4)
axes[-1,-1].set_title("Random Forest")
mglearn.discrete_scatter(X_train[:,0], X_train[:,1], y_train)
```

!["RDF"](img/random_forest1.png)


こんな感じ


```
X_train , X_test , y_train , y_test = train_test_split(cancer.data , cancer.target , random_state=0)
forest = RandomForestClassifier(n_estimators=100,random_state=0)
forest.fit(X_train,y_train)

print("Accuracy on training set : {:.3f}".format(forest.score(X_train,y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test,y_test)))

>> Accuracy on training set : 1.000
>> Accuracy on test set: 0.972
```

* 何もパラメータ調整してなくても97%の精度が出ている！
* max_features や 事前枝刈りを行うことでチューニングはできるが、大抵の場合デフォルトでも十分に機能する


!["RDF2"](img/random_forest2.png)

* ランダムフォレストの時の特徴量の重要度
* 単体の決定木よりはるかに複雑になっている


### 勾配ブースティング回帰木(勾配ブースティングマシン)

* 回帰ってあるけど、回帰にも分類にも使える


```
from sklearn.ensemble import GradientBoostingClassifier

X_train , X_test , y_train , y_test = train_test_split(cancer.data , cancer.target , random_state = 0)

gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train,y_train)

print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train,y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test,y_test)))

>> Accuracy on training set: 1.000
>> Accuracy on test set: 0.965
```
* 訓練に対して100%になっているので過剰適合している。事前枝刈りをしてやればいい。

```
gbrt = GradientBoostingClassifier(random_state=0,max_depth=1)
gbrt.fit(X_train,y_train)

print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train,y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test,y_test)))

>> Accuracy on training set: 0.991
>> Accuracy on test set: 0.972
```
* もしくは学習率を下げてやる

```
gbrt = GradientBoostingClassifier(random_state=0,learning_rate=0.01)
gbrt.fit(X_train,y_train)

print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train,y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test,y_test)))

>> Accuracy on training set: 0.988
>> Accuracy on test set: 0.965
```

* 可視化

```
gbrt = GradientBoostingClassifier(random_state=0,max_depth=1)
gbrt.fit(X_train,y_train)

plot_feature_importances_cancer(gbrt)
```

!["GBRT"](img/gradient_boost1.png)

* ランダムフォレストと比べて、に似てるが幾つかの特徴量が完全に無視されているのがわかる
* 一般的にはランダムフォレストを先に試して、精度をあげたい時とかに勾配するといいらしい


### カーネル方を用いたサポートベクタマシン

```
from sklearn.datasets import make_blobs

X,y = make_blobs(centers=4,random_state=8)
y = y % 2

mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
```

!["SVM"](img/svm1.png)

* 線形モデルのクラス分類は直線でしか分類できないので、こういうデータセットではうまくいかない

```
from sklearn.svm import LinearSVC
linear_svm = LinearSVC().fit(X,y)
mglearn.plots.plot_2d_separator(linear_svm,X)
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
```
!["SVM"](img/svm2.png)

* こいつを3次元のデータにしてやる
* やり方は feature1 ** 2 , つまり2番目の特徴量の2乗を3番目の特徴量としてやる

```
# 2番目の特徴量の2乗を追加
X_new = np.hstack([X,X[:,1:]**2])

from mpl_toolkits.mplot3d import Axes3D, axes3d
figure = plt.figure()
# 3Dで可視化

ax = Axes3D(figure , elev=-152 , azim=-26)
# y == 0の点をプロットしてから y==1の点をプロット
mask = y == 0
ax.scatter(X_new[mask,0] , X_new[mask,1] , X_new[mask,2],c='b',cmap=mglearn.cm2,s=60)
ax.scatter(X_new[~mask,0] , X_new[~mask,1] , X_new[~mask,2],c='r',marker='^',cmap=mglearn.cm2,s=60)
ax.set_xlabel=("feature 0")
ax.set_ylabel=("feature 1")
ax.set_zlabel=("feature1 ** 2")

```

!["SVM"](img/svm3.png)

* 3Dになった！


```
linear_svm_3d = LinearSVC().fit(X_new,y)
coef , intercept = linear_svm_3d.coef_.ravel() , linear_svm_3d.intercept_

figure = plt.figure()
ax = Axes3D(figure , elev=-152 , azim=-26)
xx = np.linspace(X_new[:,0].min() - 2 , X_new[:,0].max() + 2 , 50)
yy = np.linspace(X_new[:,1].min() - 2 , X_new[:,1].max() + 2 , 50)
XX,YY = np.meshgrid(xx,yy)
ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
ax.plot_surface(XX , YY , ZZ , rstride = 8 , cstride = 8 , alpha = 0.3)
ax.scatter(X_new[mask,0],X_new[mask,1],X_new[mask,2],c='b',cmap=mglearn.cm2,s=60)
ax.scatter(X_new[~mask,0],X_new[~mask,1],X_new[~mask,2],c='r',marker='^',cmap=mglearn.cm2,s=60)

ax.set_xlabel("feature0")
ax.set_ylabel("feature1")
ax.set_zlabel("feature1 ** 2")
```

!["SVM"](img/svm4.png)

決定境界はこう

!["SVM"](img/svm5.png)

線形SVMなのに線形じゃなくて楕円になっている

### カーネルトリック

* こんな感じで、非線形のデータを特徴量に加えると、線形モデルが強力になる（線形では解決できない分類を解決できる）
* ただ、実際のデータではどのデータを特徴量に加えれば良いとかわからん。(100次元とかのデータ全部とかやると計算量が多すぎる)
* こういう時、実際には計算させずに高次元空間のクラス分類機を学習させる数学的トリックが「カーネルトリック」

* ロジックは難しいけど、要は線形モデルで解決できない問題を解決する良い感じの方法


```
from sklearn.svm import SVC
X,y = mglearn.tools.make_handcrafted_dataset()
svm = SVC(kernel='rbf',C=10,gamma=0.1).fit(X,y)
mglearn.plots.plot_2d_separator(svm,X,eps=.5)
mglearn.discrete_scatter(X[:,0],X[:,1],y)
# サポートベクタをプロットする
sv = svm.support_vectors_
# サポートベクタのクラスラベルはdual_coef_の正負によって与えられる
sv_labels = svm.dual_coef_.ravel() > 0
mglearn.discrete_scatter(sv[:,0],sv[:,1],sv_labels,s=15,markeredgewidth=3)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

```

!["SVM"](img/svm6.png)

* RBFカーネル法を用いたSVMによる決定境界とサポートベクタ
* 境界は滑らかで、非線形。ここではCとgammaの2つのパラメータで調整している


#### SVMパラメータの調整

* gammaパラメータはガウシアンカーネルの幅を調整する。このパラメータが点が近いという事を意味するスケールを決定する
* Cパラメータは線形モデルで用いられたのと同様の正則化パラメータである。個々のデータポイントの重要度を制限する(dual_coef_を制限する)
* これらのパラメータを変化させるとどうなるか・・・


```
fig,axes = plt.subplots(3,3,figsize=(15,10))

for ax,C in zip(axes ,[-1,0,3]):
    for a, gamma in zip(ax,range(-1,2)):
        mglearn.plots.plot_svm(log_C=C,log_gamma=gamma,ax=a)

axes[0,0].legend(["class 0","class 1","sv class 0","sv class 1"],ncol=4,loc=(.9,1.2))

```

!["SVM"](img/svm7.png)

* 左から右へパラメータgammaを0.1〜10に変化させている。
* gammaが小さいとガウシアンカーネルの直径が大きくなり、多くの点を近いと判断している
* 左にいくにつれて決定境界が滑らかになり、右に行くにつれて個々のデータポイントをより重視いｓている
* gammaが小さいと = 決定境界はゆっくりと変化せず、モデルの複雑さは小さくなる
* 上から下へパラメータCを0.1から1000に変化させている
* 線形モデルと同様、Cが小さいと個々のデータポイントが与える影響は少なくなる
* Cを大きくするとこれらのデータポイントがより強い影響を持ち、正しくクラス分類されるに決定境界を曲げる

RBFカーネル法を用いたSVMをcancerデータセットに適応してみる
デフォルトはC=1,gamma=1/n_featuresになっている

```
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,random_state=0)

svc = SVC()
svc.fit(X_train,y_train)

display(svc.score(X_train,y_train))
display(svc.score(X_test,y_test))

>> 1.0
>> 0.6293706293706294

```

* 訓練セットは100%でテストセットでは63%になっているので、過剰適合している
* SVMはうまく動く場合が多いが、パラメータの設定とデータのスケールに敏感な問題があるA
* 特に全ての特徴量の変位が同じスケールである事を要求する

#### SVMのためのデータ前処理

* 特徴量をだいたいおんなじスケールにするためのものが用意されている
* MinMaxScalerってのがあるけど、ここでは手で実装してみる

```
# 訓練セットごとの特徴量ごとに最小値を計算
min_on_training = X_train.min(axis=0)
# 訓練セットの特徴量ごとにレンジ(最大値-最小値)を計算
range_on_training = (X_train - min_on_training).max(axis=0)

# 最小値を引いてレンジで割る
# 個々の特徴量はmin=0,max=1となる
X_train_scaled = (X_train - min_on_training) / range_on_training
display(X_train_scaled.min(axis=0))
display(X_train_scaled.max(axis=0))

>> array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
>> array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

```


```
# テストセットにも同じ変換をする
X_test_scaled = (X_test - min_on_training) / range_on_training

svc = SVC()
svc.fit(X_train_scaled,y_train)

display(svc.score(X_train_scaled,y_train))
display(svc.score(X_test_scaled,y_test))

>> 0.9483568075117371
>> 0.951048951048951
```
* スケール変更によって精度があがっている。
* Cをいじってみる

```
svc = SVC(C=1000)
svc.fit(X_train_scaled,y_train)
display(svc.score(X_train_scaled,y_train))
display(svc.score(X_test_scaled,y_test))
>> 0.9882629107981221
>> 0.972027972027972
```
* さらに精度が上がった

#### メリット・デメリット

* データにわずかな特徴量しかない場合にも複雑な決定境界を生成できる
* 低次元のデータでも高次元のデータでも(つまり特徴量が多くても少なくても)うまく機能する
* ただし、サンプルの個数が大きくなるとうまく機能しない
    * だいたい10,000くらいまでは大丈夫。100,000サンプルくらいになると、メモリ使用量などで難しくなる
* 前処理とパラメータ調整がほぼ必要なのでたいへん
    * なので勾配ブースティングランダムフォレストとかに比べたらめんどい
* 特徴量が似た測定器の測定結果(カメラのピクセルとか)など、同じスケールになる場合には試してみる価値がある


## ニューラルネットワーク（ディープラーニング）

* ニューラルネットワークというアルゴリズムが「ディープラーニング」と呼ばれて最近注目を集めている
* ここでは比較的簡単な多重パーセプトロン(multilayer perceptron:MLP)によるクラス分類と回帰についてだけ議論する

* two moonsをMLPでやってみる

```
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons

X,y = make_moons(n_samples=100,noise=0.25,random_state=3)
X_train , X_test , y_train , y_test = train_test_split(X,y,stratify=y,random_state=42)

mlp = MLPClassifier(solver='lbfgs',random_state=0).fit(X_train,y_train)
mglearn.plots.plot_2d_separator(mlp,X_train,fill=True,alpha=.3)
mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
```

!["MLP"](img/MLP.png)

* ニューラルネットワークは比較的なめらかな決定境界を学習している
* デフォルトではMLPは100隠れユニットを用いる
    * これは小さいデータセットに対しては明らかに多すぎる
    * この数を減らし、モデルの複雑さを減らしてもいい結果が出る


```
mlp = MLPClassifier(solver='lbfgs',random_state=0,hidden_layer_sizes=[10])
mlp.fit(X_train,y_train)
mglearn.plots.plot_2d_separator(mlp,X_train,fill=True,alpha=.3)
mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
```

* 隠れユニット10の決定境界

!["MLP"](img/MLP2.png)

* 隠れユニット数を減らすと滑らかさを失う。

* さらに、複雑さをリッジ回帰や線形クラス分類器で行ったのと同様にl2ペナルティで重みを0に近づける事で制御できる
* MLPClassifierではこのパラメータは線形回帰モデルと同じくalphaで、デフォルトでは非常に小さい値（弱い正則化）に設定されている
* 10ユニット、もしくは100ユニットの2層隠れ層をもつニューラルネットワークをtwo_moonsデータセットに適用した時のパラメータalphaの効果は次の感じ

```
fig,axes = plt.subplots(2,4,figsize=(20,8))
for axx , n_hidden_nodes in zip (axes,[10,100]):
    for ax,alpha in zip(axx,[0.0001,0.01,0.1,1]):
        mlp = MLPClassifier(solver='lbfgs' , random_state=0,hidden_layer_sizes=[n_hidden_nodes,n_hidden_nodes],alpha=alpha)
        mlp.fit(X_train,y_train)
        mglearn.plots.plot_2d_separator(mlp,X_train,fill=True,alpha=.3,ax=ax)
        mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train,ax=ax)
        ax.set_title("n_hidden=[{},{}] \n alpha={:.4f}".format(n_hidden_nodes,n_hidden_nodes,alpha))
```


!["MLP"](img/MLP3.png)

* ニューラルネットワークにはたくさんのパラメータがあり、それらに乱数で重みをつける
* なので乱数シードが違うと全然違うモデルができたりする

* 全部同じパラメータだけど乱数シードだけが違う例

```
fig,axes = plt.subplots(2,4,figsize=(20,8))
for i,ax in enumerate(axes.ravel()):
    mlp = MLPClassifier(solver='lbfgs',random_state=i,hidden_layer_sizes=[100,100])
    mlp.fit(X_train,y_train)
    mglearn.plots.plot_2d_separator(mlp,X_train,fill=True,alpha=.3,ax=ax)
    mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train,ax=ax)
```

!["MLP"](img/MLP4.png)


* cancerデータでいろいろする
```
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer data per-feature maxima:\n{}".format(cancer.data.max(axis=0)))

>> cancer data per-feature maxima:
[2.811e+01 3.928e+01 1.885e+02 2.501e+03 1.634e-01 3.454e-01 4.268e-01
 2.012e-01 3.040e-01 9.744e-02 2.873e+00 4.885e+00 2.198e+01 5.422e+02
 3.113e-02 1.354e-01 3.960e-01 5.279e-02 7.895e-02 2.984e-02 3.604e+01
 4.954e+01 2.512e+02 4.254e+03 2.226e-01 1.058e+00 1.252e+00 2.910e-01
 6.638e-01 2.075e-01]
```
* デフォでやる
```
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,random_state=0)

mlp = MLPClassifier(random_state=42)
mlp.fit(X_train,y_train)

print("Accuracy on training set : {:.2f}".format(mlp.score(X_train,y_train)))
print("Accuracy on test set : {:.2f}".format(mlp.score(X_test,y_test)))

>> Accuracy on training set : 0.94
>> Accuracy on test set : 0.92
```

* MLPの精度はデフォでも結構いいが、他のモデルほどでもない。
* MLPもデータの特徴量が同じ範囲に収まっている事を前提にしている
* 次に StandardScaler を手で実装してみる

```
# 訓練セットの特徴量ごとの平均値を算出
mean_on_train = X_train.mean(axis=0)
# 訓練セットの特徴量ごとの標準偏差を算出
std_on_train = X_train.std(axis=0)

# 平均を引き、標準偏差の逆数でスケール変換する
# これで mean=0 , std=1 になる
X_train_scaled = (X_train - mean_on_train) / std_on_train
# 全く同じ変換をテストセットにもやる
X_test_scaled = (X_test - mean_on_train) / std_on_train

mlp = MLPClassifier(random_state=0)
mlp.fit(X_train_scaled,y_train)

print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled,y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled,y_test)))

>> Accuracy on training set: 0.991
>> Accuracy on test set: 0.965
>> ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
```

* スケール変換でめっちゃよくなった
* 最後の警告は「繰り返し回数が上限に達したが、最適化が収束していない」とある

```
mlp = MLPClassifier(max_iter=1000, random_state=0)
mlp.fit(X_train_scaled,y_train)

print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled,y_train)))
print("Accuracy on ttestraining set: {:.3f}".format(mlp.score(X_test_scaled,y_test)))

>> Accuracy on training set: 1.000
>> Accuracy on ttestraining set: 0.972
```

* 繰り返し回数を増やすと、訓練セットに対する効果は上がったが、汎化性能は上がっていない（って本に書いてあるけどちょっと上がってる）
* 次にalphaパラメータを0.0001から1にあげて重みに関して正則化を強化してみる

```
mlp = MLPClassifier(max_iter=1000,alpha=1,random_state=0)
mlp.fit(X_train_scaled,y_train)

print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled,y_train)))
print("Accuracy on ttestraining set: {:.3f}".format(mlp.score(X_test_scaled,y_test)))

>> Accuracy on training set: 0.988
>> Accuracy on ttestraining set: 0.972
```

* こうすると、ベストの性能になった
    * ※「ベストな性能」は今まで扱ったどのモデルでも0.972になっている（ほんと？）
    * これは、これらのモデルがまったく同じ数、4つの点のクラス分類を誤っているという事を意味する
    * これはデータセットが小さいせいかもしれないし、間違う4つのデータポイントが残りのデータポイントからかけ離れているせいかもしれない


* ニューラルネットワークが学習した内容を解析する事はできる
* そのうちの1つはモデル内部の重みを見ること

```
plt.figure(figsize=(20,5))
plt.imshow(mlp.coefs_[0],interpolation='none',cmap='viridis')
plt.yticks(range(30),cancer.feature_names)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()
```

!["MLP"](img/MLP5.png)

* 明るいところが大きな正の値、暗いところが大きな負の値
* めちゃめちゃ見辛い


#### メリットデメリット

* scikit-learnはニューラルネットワークでできることの一部しかカバーしていない
* 本当に使う場合はkeras , lasagna , tensor-flow とかがいけてるんやで
* それらはGPUをサポートしていて、これを使うと10〜100倍もはやくなるんや
* ニューラルネットワーク最大の利点は大量のデータを使って、めちゃくちゃ複雑なモデルを構築できること
* 十分な計算時間をかけて、データを用意して、パラメータを調整すれば他のアルゴリズムよりめっちゃ強いのができあがることが多い
* それが逆にデメリットでもある。大きくて強力なものには訓練に時間がかかるし、慎重にデータの前処理が必要
* SVMと同じく同様にデータが「同質」な場合最もよく機能するが、様々な種類の特徴量を持つデータにたいしては決定木に基づくモデルの方が性能が良い


#### ニューラルネットワークの複雑さ推定

* 実際にニューラルネットワークのパラメータを調整する一般的なやり方は
    * まず過剰適合できるように大きいネットワークを作って、タスクがそのネットワークで訓練データを学習できる事を確認する
    * 次にネットワークを小さくするか、alphaを増やして正則化を強化して汎化性能をあげていく
