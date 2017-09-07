# machine-learning-keywords










## 活性化関数
出力層→確率を出力する関数＝通常SigmoidまたはSoftmax
隠れ層→受け取る値が小さければ小さい値を出力し、受け取る値が大きければ大きい値を出力すればOK
勾配消失問題を回避するために様々な活性化関数が使われるようになった。
 


### ステップ関数


### Sigmoid

### tanh 
hyperbolic tangent function（ハイパボリックタンジェント）
日本語：　双曲線正接関数
数式：

```math
tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
```


### ReLU
- Rectified Linear Unit
- 正規化線形関数、ランプ関数

### Leaky ReLU (LReLU)


### Rarametric ReLU (PReLU)

### その他のReLU派生関数
- Randomized ReLU (RReLU)
- Exponential Linear Unit (ELU)

まずReLUやLReLUを用いれば十分な場合が多い


## 学習率に関する設定手法

### 学習率 (learning rate)



### モメンタム (momentum)
学習率の値自体は変更しないが、モメンタム項と呼ばれる調整項を用いることによって、「学習率を最初は大きく、徐々に小さくする」という動きを擬似的に表現。
TensorFlowでは、tf.train.MomentumOptimizer(0.01, 0.9)
Keras


### Nesterovモメンタム
Nesterov氏が提案した、モメンタムに変更を加えたもの。「パラメータがどの方向に向かうべきか」を式に織り込んだ。

```python:TensorFlow
TensorFlowでは、tf.train.MomentumOptimizer(0.01, 0.9, use_nesterov=True)
```

### Adagrad (Adaptive Gradient Algorithm)
学習率の値そのものを更新する手法。
モメンタムを比べてハイパーパラメータが少なく、これまでの勾配に基づき自動で学習率を修正するため、より扱いやすい手法といえる。

```python:TensorFlow
TensorFlow　tf.train.AdagradOptimizer(0.01)
```

### Adadelta
Adagrad=学習のステップを経る毎に勾配にかかる係数の値が急激に小さくなってしまい、学習が進まなくなるという問題がおこる。
この問題を解決したのがAdadelta。

```python:TensorFlow
optimizer = tf.train.AdadeltaOptimezer(leraning_rate=1.0, rho=0.95
```

### RMSprop
Adadelta同様、Adagradにおける学習率の急激な減少の問題を解決するための手法。
論文にはなっておらず、Courseraのスライドでまとめられている。
Adadeltaの簡易版といえるもの。

```python:TensorFlow
optimizer = tf.train.RMSPoropOptimizer(0.001)
```

### Adam (Adaptive Moment estimation)
AdadeltaやRMSpropと同様、Adagradにおける学習率の急激な減少の問題を解決するための手法。
AdadeltaやRMSprop　直前のステップt-1までの勾配の2乗の移動平均  `vt:=E[g^2]t` を指数関数的に減衰
Adam　単純な勾配の移動平均  `mt:= E[g]t` を指数関数的に減衰

```python:TensorFlow
optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
```




## オーバーフィッティングを回避する手法

### ドロップアウト (dropout)
学習の際にニューロンをランダムに除外(=ドロップアウト)させるもの。
ドロップアウト確率pは一般的にはp=0.5
学習後のテスト・予測の際はドロップアウトしない。代わりに例えば重みがWだとした場合、学習全体の平均である(1-p)Wを出力に用いる。
ドロップアウトは疑似的にアンサンブル学習をしている状態。



### Early Stopping
学習回数（＝エポック数）を適切に設定する手法の１つ。基本的な考え方は「前のエポックのときと比べて誤差が増えたら学習を打ち切る」。
ただし、直前のエポックの誤差のみで比較すればよいわけではない。（特にドロップアウトを適用する場合、まだ学習が行われていないニューロンが存在しうるため）
従って「ある一定のエポック数を通してずっと誤差が増えていたら学習を打ち切る」ようにする。





## データの加工と重みの初期化

### 正規化 （せいきか）(normalization)



### 白色化 (whitening)
平均：0、　分散：1
特徴成分が非相関となるように正規化する
画像処理 

### GCN (Global Contrast Normalization)


### 正則化　（せいそくか）(regularization)


### 標準化　(standardization)



### Batch Normalization
ミニバッチ毎に正規化を実施する。（単純な正規化はデータセットを事前に正規化する処理）
Batch Normalizationの利点として、「学習率を大きくしても学習がうまくいく」「ドロップアウトを用いなくても汎化性能が高い」と文献にある。


### 重みの初期化手法
- LeCun et al.1988　...正規分布あるいは一様分布による初期化。一様分布の場合のコードは下記。

```python:Numpy
np.random.uniform(low=-np.sqrt(1.0 / n),
                  high=np.sqr(1.0 / n),
                  size=shape)
```

```python:Keras
kernel_initializer='lecun_uniform'
```

- Glorot and Bengio 2010　...正規分布および一様分布を用いる場合の手法

一様分布の場合

```python:Numpy
np.random.uniform(low=-np.sqrt(6.0 / (n_in + n_out)),
                  high=np.sqr(1.0 / (n_in + n_out)),
                  size=shape)
```

正規分布の場合

```python:Numpy
np.sqrt(3.0 / (n_in + n_out)) * np.random.normal(size=shape)
```


- He et al. 2015　...ReLUを用いる場合の初期化について

```python:Numpy
np.sqrt(2.0 / n) * np.random.normal(size=shape)
```



## その他

### アンサンブル学習 (ensemble learning)



