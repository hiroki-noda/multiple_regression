# csvを読み込んだり、データセットを操作するときに使うライブラリ
import pandas as pd

# 配列を扱うライブラリ
import numpy as np

# scikit-learnに元々収録されているデータセットをインポート
from sklearn import datasets

# scikit-learnで利用できる回帰モデル
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# Pythonの実行時に引数を指定することができる
import argparse

# 回帰アルゴリズムと、使用するデータセットを引数で指定できるようにする
parser = argparse.ArgumentParser()
parser.add_argument(
    '--algo', default='Least_Squares', help='回帰アルゴリズム')
parser.add_argument(
    '--dataset', default='wine', help='データセット')
args = parser.parse_args()

#データセットの準備
# https://scikit-learn.org/stable/datasets.html
# https://note.nkmk.me/python-sklearn-datasets-load-fetch/
if args.dataset == "wine":
    # wget http://pythondatascience.plavox.info/wp-content/uploads/2016/07/winequality-red.csv でダウンロード可能
    df = pd.read_csv('winequality-red.csv', sep=';')
    #x:入力データ
    x = df[['density', 'volatile acidity']].values
    #y:予測対象のデータ
    y = df[['alcohol']].values
elif args.dataset == "boston":
    # wget http://lib.stat.cmu.edu/datasets/boston でダウンロード可能
    df = pd.read_csv('boston', sep="\s+", skiprows=22, header=None)
    x = np.hstack([df.values[::2, :], df.values[1::2, :2]])
    y = df.values[1::2, 2]
elif args.dataset == "diabates":
    data = datasets.load_diabetes()
    x = data.data
    y = data.target
elif args.dataset == "linnerud":
    data = datasets.load_linnerud()
    x = data.data
    y = data.target

#データセットを8:2の割合で訓練用データとテスト用データにわける
train_size = int(0.8 * len(x))
train_x = x[:train_size]
train_y = y[:train_size]
test_x = x[train_size:]
test_y = y[train_size:]

# 回帰モデルの設定
# 公式ドキュメントによると、他にも様々な回帰モデルが存在します
# https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
if args.algo == "Least_Squares":
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression
    model = linear_model.LinearRegression()
elif args.algo == "Ridge":
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
    model = linear_model.Ridge(alpha=0.5)
elif args.algo == "SGD":
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor
    model = linear_model.SGDRegressor(max_iter=500)
elif args.algo == "SVR":
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR
    model = SVR()
elif args.algo == "MLP":
    # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
    model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, activation='relu', solver='adam', learning_rate_init=0.001)

# 回帰モデルのフィッティング
model.fit(train_x, np.squeeze(train_y))

# テストデータを用いてモデルの評価
predict_y = model.predict(test_x)
# 実際の値との二乗誤差を出力
print(np.linalg.norm(np.squeeze(test_y) - predict_y))