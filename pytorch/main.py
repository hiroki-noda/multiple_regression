#csvからデータを読み込むときに使うライブラリ
import pandas as pd
#配列を扱うライブラリ
import numpy as np
#深層学習ライブラリ
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#プログラム実行時に入力を受け付けるためのライブラリ
import argparse
#ニューラルネットの部分だけ別のファイルにあるため、それをインポートする
from newral_net import Net

from sklearn import datasets
from sklearn.metrics import mean_squared_error

#ハイパーパラメータの部分は入力で指定できるようにする
parser = argparse.ArgumentParser()
parser.add_argument(
    '--hidden-size', type=int, default=32, help='中間層のノード数')
parser.add_argument(
    '--lr', type=float, default=1e-4, help='学習率')
parser.add_argument(
    '--batch-size', type=int, default=10, help='ミニバッチのサイズ')
parser.add_argument(
    '--max-epoch', type=int, default=500, help='エポック数（何回学習を回すか）')
parser.add_argument(
    '--dataset', default='wine', help='データセット')
args = parser.parse_args()


#データセットの準備
# https://scikit-learn.org/stable/datasets.html
# https://note.nkmk.me/python-sklearn-datasets-load-fetch/

# 入力データxは（データ数×次元数）
# 予測対象データyは（データ数×１）or（データ数）
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

#pytorchではtensorと呼ばれる型にデータを変換する必要がある
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

#生のデータでは、1599個のxと、それに対応するyがセットになっている
#この1599セットを8:2の割合で訓練用データとテスト用データにわける
train_size = int(0.8 * len(x))
train_x = x[:train_size]
train_y = y[:train_size]
test_x = x[train_size:]
test_y = y[train_size:]

# TODO:データの正規化

train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
test_dataset = torch.utils.data.TensorDataset(test_x, test_y)

#ミニバッチのサイズ
#1000以上ある訓練用データのうち、何個ずつニューラルネットに入れるかを指定している
batch_size = args.batch_size

#data_loaderを用いることで、ランダムにシャッフルされた訓練用データから
#ミニバッチのサイズ分だけ入力データxとそれに対応する教示データyを出力してくれるようになる
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size)

#ニューラルネットワークの定義
net = Net(x.size()[1], args.hidden_size, 1)
#誤差関数の定義
#今回はニューラルネットワークの出力をある値に近づけるように学習させるので、二乗誤差を用いる
loss_function = nn.MSELoss()
#最適化関数
optimizer = optim.Adam(net.parameters(), lr=args.lr)

max_epoch = args.max_epoch
for epoch in range(max_epoch):
    for batch in train_loader:
        # dataloaderからミニバッチを取り出す
        x, y = batch
        # 勾配の初期化
        optimizer.zero_grad()
        # xをニューラルネットワークに代入し、yを推定する
        y_hat = net(x)
        # 損失関数による誤差の計算
        loss = loss_function(y_hat.squeeze(), y.squeeze())
        # 誤差逆伝播
        loss.backward()
        # 重みの更新
        optimizer.step()

    # テストデータを用いて学習状況を確認
    with torch.no_grad():
        test_loss = []
        for batch in test_loader:
            x, y = batch
            y_hat = net(x)
            loss = loss_function(y_hat.squeeze(), y.squeeze())
            test_loss.append(loss.item())
        print('test_loss: ', np.array(test_loss).mean())

with torch.no_grad():
    predict_y= net(test_x)
    print("Mean Square Error = {}".format(mean_squared_error(np.squeeze(test_y), predict_y)))


