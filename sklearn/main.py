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

# 平均二乗誤差の関数
from sklearn.metrics import mean_squared_error

# Pythonの実行時に引数を指定することができる
import argparse

# 回帰アルゴリズムと、使用するデータセットを引数で指定できるようにする
parser = argparse.ArgumentParser()
parser.add_argument(
    '--algo', default='Least_Squares', help='回帰アルゴリズム')
parser.add_argument(
    '--dataset', default='wine', help='データセット')
parser.add_argument(
    '--plot', action='store_true', default=False, help='データを三次元グラフにプロットする')
args = parser.parse_args()

def main():
    #データセットの準備
    # https://scikit-learn.org/stable/datasets.html
    # https://note.nkmk.me/python-sklearn-datasets-load-fetch/
    if args.dataset == "wine":
        # wget http://pythondatascience.plavox.info/wp-content/uploads/2016/07/winequality-red.csv でダウンロード可能

        # csvファイルの読み込み
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
    elif args.dataset == "california":
        data = datasets.fetch_california_housing()
        x = data.data
        y = data.target
    else:
        raise NotImplementedError

    # この時点で、
    # x:（データ数×次元数）
    # y:（データ数×１）or（データ数）
    # の配列になっていればよい

    # データセットを8:2の割合で訓練用データとテスト用データにわける
    # 訓練用データとテスト用データが元々別に用意されている場合は、この操作は不要
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
    else:
        raise NotImplementedError
    
    # yの形が（データ数×１）の場合、（データ数）に成形する
    train_y = np.squeeze(train_y)

    # 回帰モデルのフィッティング
    model.fit(train_x, train_y)

    # テストデータを用いてモデルの評価
    predict_y = model.predict(test_x)
    # 実際の値との平均二乗誤差を出力
    print("Mean Square Error = {}".format(mean_squared_error(np.squeeze(test_y), predict_y)))
    # グラフにプロット
    if args.plot:
        plot(test_x, test_y, predict_y)

# xのランダムな２軸と、yを三次元にプロット
def plot(test_x,test_y,predict_y):
    import matplotlib.pyplot as plt
    import random
    dim = random.sample(range(0,test_x.shape[1]), k=2)
    x1 = test_x[:,dim[0]]
    x2 = test_x[:,dim[1]]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #正解データ
    ax.scatter(x1, x2, test_y, label="Target")
    # モデルによって予測したデータ
    ax.scatter(x1, x2, predict_y, label="Predicted")
    # 軸ラベルを設定
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.legend()
    # グラフの表示
    plt.show()

if __name__ == "__main__":
    main()
