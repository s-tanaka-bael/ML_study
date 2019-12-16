# titanicにチャンレジする

titanicチャレンジ

https://www.kaggle.com/c/titanic/


### いきなりcsvインポートでハマる

```
import pandas as pd

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]

>> FileNotFoundError: [Errno 2] File b'../input/train.csv' does not exist: b'../input/train.csv'

```

どうもディレクトリを掘るとそこを見つけられないっぽい。
import os とかしてos.pathgetcwd()すると、全然違うディレクトリで実行されているっぽい。

```
import os
os.getcwd()

>> '/Users/shinyatanaka/Desktop/unity_study/chap6/New Unity Project (1)'

```

なんやそれ。実行中のファイルをとる `__file__` はnotebookだとつかえないくさい。ダル。

```
# data analysis and wrangling
import pandas as pd
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]

>> FileNotFoundError: [Errno 2] File b'train.csv' does not exist: b'train.csv'

```
ファイル移動して実行ファイルと同じにしてもあかんかった。なんやそれ