# mlflow_keras_example
MLflowを使ったkerasプログラム

# Overview
kerasを使ったMLflowの実験プログラム。

Usageを要参照。

コード内に問題がある可能性があるので，もしお気づきになられたら修正 or 連絡をお願い致します．

# Environment
- MacOS Mojave 10.14.5
- Python 3.7.0

```
$ git clone https://github.com/T-Sumida/mlflow_keras_example.git
$ cd mlflow_keras_example
$ pip install -r requirements.txt
```

# Usage
## プログラムの実行
```
usage: mlflow_example.py [-h] [-d DATASET] [-m MODEL] [-lr LR] [-e E] [-b B]
                         exp_name

MLflowを試すプログラム

positional arguments:
  exp_name              MLflowに与える実験名[mnist_cnn とか fashion_fullyとか]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        データセットを指定[mnist or fashion]
  -m MODEL, --model MODEL
                        モデルを指定[fully or cnn]
  -lr LR                学習率を指定
  -e E                  エポック数
  -b B                  バッチサイズを指定

（例）
$python mlflow_example.py test_experiment -d mnist -m cnn -lr 0.001 -e 10 -b 128
Epoch 1/10
48000/48000 [==============================] - 45s 937us/step - loss: 0.5244 - acc: 0.8086 - val_loss: 0.3597 - val_acc: 0.8689
Epoch 2/10
48000/48000 [==============================] - 44s 925us/step - loss: 0.3116 - acc: 0.8866 - val_loss: 0.2950 - val_acc: 0.8926
Epoch 3/10
48000/48000 [==============================] - 47s 976us/step - loss: 0.2553 - acc: 0.9066 - val_loss: 0.2585 - val_acc: 0.9047
Epoch 4/10
48000/48000 [==============================] - 39s 809us/step - loss: 0.2245 - acc: 0.9169 - val_loss: 0.2499 - val_acc: 0.9093
Epoch 5/10
48000/48000 [==============================] - 34s 716us/step - loss: 0.1984 - acc: 0.9271 - val_loss: 0.2323 - val_acc: 0.9173
```

# License
Copyright © 2019 T_Sumida Distributed under the MIT License.
