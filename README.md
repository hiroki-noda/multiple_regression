# Multiple Regression
## 必要ライブラリ  
* numpy  
* pandas  
* scikit-learn  
* pytorch（任意）  
(https://pytorch.org/get-started/locally/)  

## データセット  
### scikit-learnに収録されているデータセット  
* diabetes(糖尿病の進行状況)  
* linnerud(生理学的測定結果と運動測定結果)  
* california(カリフォルニアの住宅価格)  
参考  
https://scikit-learn.org/stable/datasets.html  
https://note.nkmk.me/python-sklearn-datasets-load-fetch/  
### webからダウンロードしたデータセット  
* wine（赤ワインの品質）  
* boston（ボストンの住宅価格）  
### オリジナルのデータセットを使う場合  
winequality-red.csvを参考に、csvファイルをpandasで読み込むのがよいでしょう  
X = (x_1, x_2, ..., x_n) とyの組がNセットあるようなデータセットを用いる場合、  
入力データxは(N×n)  
予測対象データyは(N×1)or(N)      
という形のnumpy配列であれば大丈夫です  

## アルゴリズム
scikit-learnに収録されている回帰モデルなら基本的に使えるはずです  
https://scikit-learn.org/stable/supervised_learning.html#supervised-learning  
### 線形回帰
* 最小2乗回帰
* リッジ回帰
* 確率的勾配降下法（線形モデル）
### 非線形回帰
* サポートベクター回帰
* マルチレイヤーパーセプトロン(MLP)＝ニューラルネットワーク  

※他にもたくさんあるのでいろいろ追加してみてください  
※pytorchバージョンも作りましたが、基本的にscikit-learnのMLPで事足りると思うので必要ないと思います

## 実行例
sklearn  
`python sklearn/main.py --algo SVR --dataset boston`  
pytorch  
`python pytorch/main.py --hidden-size 64 --lr 5e-4 --batch-size 10 --dataset diabates` 

`--plot`とすると、xのランダムな２軸と、y（正解値、予測値）を三次元にプロットしたグラフを出力します  
![result](https://user-images.githubusercontent.com/41198657/179135411-96a9f8ab-eba8-4c97-98fa-1e5d572937db.png)
