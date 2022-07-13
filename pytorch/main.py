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
args = parser.parse_args()

#csvデータの飲み込み
df = pd.read_csv('winequality-red.csv', sep=';')
#入力データ（２次元）
x = df[['density', 'volatile acidity']]
#予測対象のデータ（１次元）
y = df[['alcohol']]

#pytorchではtensorと呼ばれる型にデータを変換する必要がある
x = torch.tensor(x.values, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.float32)

#生のデータでは、1599個のxと、それに対応するyがセットになっている
#この1599セットを8:2の割合で訓練用データとテスト用データにわける
train_size = int(0.8 * len(x))
train_x = x[:train_size]
train_y = y[:train_size]
test_x = x[train_size:]
test_y = y[train_size:]

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
net = Net(2,args.hidden_size,1)
#誤差関数の定義
#今回はニューラルネットワークの出力をある値に近づけるように学習させるので、二乗誤差を用いる
loss_function = nn.MSELoss()
#最適化関数
#Adamは広く使われている関数で、学習が進むにつれて学習率を減衰させるような仕組みになっている
optimizer = optim.Adam(net.parameters(), lr=args.lr)

max_epoch = args.max_epoch
for epoch in range(max_epoch):
    for batch in train_loader:
        x, y = batch
        
        optimizer.zero_grad()
        y_hat = net(x)
        
        loss = loss_function(y_hat, y)
        
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_loss = []
        for batch in test_loader:
            x, y = batch
            y_hat = net(x)
            loss = loss_function(y_hat, y)
            test_loss.append(loss.item())
        print('test_loss: ', np.array(test_loss).mean())

with torch.no_grad():
    predict = net(test_x)
    print(np.linalg.norm(test_y.numpy() - predict.numpy()))


