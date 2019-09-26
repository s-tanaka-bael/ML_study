# Pythonで始める機械学習

## 1章はじめに

### Irisデータでの練習メモ


#### 基本的なpythonのメモ

ライブラリのインポート
```
from sklearn.datasets import load_iris
```
のようなコードはskleran.datasetsの下にあるload_irisをインポートしているだけ。
単純なインポートはimportでおｋ

---
表示させる系
```
display(iris_dataset.keys())
```
displayする

---

配列の範囲
```
display(iris_dataset[data][:100])
```
コロンで配列の要素の範囲指定ができる。
↑だと0〜100番目までが出る。

---

配列の要素数
```
display(iris_dataset[data].shape)
```
配列の要素数がでる。(150,4)みたいな

---

#### sklearnのメソッド

#### train_test_split

```
from sklearn.model_selection import train_test_split
X_train , X_test , y_train, y_test = train_test_split(
    iris_dataset['data'] , iris_dataset['target'] , random_state = 0)
```
train_test_splitでデータセットを訓練セット、テストセットに分けられる。75%を訓練にして、25%をテストにしてくれる。random_stateは固定しておく方が良い。

第一引数はデータ、第二引数はラベル(データがそれぞれどの分類になるのか)




. 