# coding:utf-8

import os
import argparse
from typing import Tuple
import numpy as np
import mlflow
import mlflow.keras
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist, fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 引数でデータセット・モデル・パラメータを変更できるようにしておく
parser = argparse.ArgumentParser(description="MLflowを試すプログラム")
parser.add_argument('exp_name', type=str,
                    help="MLflowに与える実験名[mnist_cnn とか fashion_fullyとか]")
parser.add_argument('-d', '--dataset', type=str,
                    default='mnist', help="データセットを指定[mnist or fashion]")
parser.add_argument('-m', '--model', type=str,
                    default='fully', help="モデルを指定[fully or cnn]")
parser.add_argument('-lr', type=float,
                    default=1e-3, help="学習率を指定")
parser.add_argument('-e', type=int,
                    default=20, help="エポック数")
parser.add_argument('-b', type=int,
                    default=128, help="バッチサイズを指定")
args = parser.parse_args()


def get_datasets() -> Tuple[Tuple[np.array, np.array], Tuple[np.array, np.array], Tuple[np.array, np.array]]:
    """データセットを取得する

    Returns:
        (np.array, np.array), (np.array, np.array), (np.array, nparray) -- \
         (train_x, train_y), (val_x, val_y), (test_x, test_y)
    """
    if args.dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif args.dataset == 'fashion':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        print('--dataset Error')
        print('dataset = [mnist or fashion]')
        exit(1)

    # 学習データの２割を検証データとして用いる
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2)

    # ラベルデータをone-hotに変換しておく
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    y_test = to_categorical(y_test)

    # 画素データを0~1の範囲にリスケールする
    x_train = x_train.astype('float32') / 255
    x_val = x_val.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def build_fully_model(input_num: int, output_class_num: int, weight_path: str = None) -> keras.Model:
    """全結合モデルを作成する

    Arguments:
        input_num {int} -- 入力特徴量次元
        output_class_num {int} -- 出力クラス数

    Keyword Arguments:
        weight_path {str} -- モデルの重みパス (default: {None})

    Returns:
        keras.model -- 全結合モデル
    """
    model = Sequential()
    model.add(Dense(512, activation='relu',
                    input_shape=(input_num,), name="dense1"))
    model.add(Dropout(0.2, name='dropout1'))
    model.add(Dense(512, activation='relu', name='dense2'))
    model.add(Dropout(0.2, name='dropout2'))
    model.add(Dense(output_class_num, activation='softmax', name='last_dense'))
    if weight_path is not None:
        model.load_weights(weight_path)
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=args.lr),
                  metrics=['accuracy'])
    return model


def build_cnn_model(
    width: int, height: int, ch: int, output_class_num: int, weight_path: str = None) -> keras.Model:
    """CNNモデルを作成する

    Arguments:
        width {int} -- 入力画像の幅
        height {int} -- 入力画像の高さ
        ch {int} -- 入力画像のチャンネル数
        output_class_num {int} -- 出力クラス数

    Keyword Arguments:
        weight_path {str} -- モデルの重みパス (default: {None})

    Returns:
        keras.model -- CNNモデル
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu',
                     name='conv1', input_shape=(height, width, ch)))
    model.add(MaxPooling2D((2, 2), name='pool1'))

    model.add(Conv2D(64, (3, 3), padding='same', name='conv2'))
    model.add(MaxPooling2D((2, 2), name='pool2'))

    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu', name='conv3'))

    model.add(Flatten(name='flatten'))
    model.add(Dense(64, activation='relu', name='dense4'))
    model.add(Dense(10, activation='softmax', name='last_dense'))
    if weight_path is not None:
        model.load_weights(weight_path)
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=args.lr),
                  metrics=['accuracy'])
    return model


def calc_metrics(model: keras.Model, X: np.array, Y: np.array) -> Tuple[float, float, float, float]:
    """メトリクスを計算する

    Arguments:
        model {keras.model} -- 推論用モデル
        X {numpy.array} -- 入力データ
        Y {numpy.array} -- ラベルデータ（one-hot形式）

    Returns:
        [float, float, float, float] -- [accuracy, precision, recall, f1]
    """
    pred_y = []
    true_y = []
    for z in zip(X, Y):
        (x, y) = z
        x = np.expand_dims(x, axis=0)
        result = model.predict(x)
        pred_y.append(np.argmax(result))
        true_y.append(np.argmax(y))

    acc = accuracy_score(true_y, pred_y)
    prec = precision_score(true_y, pred_y, average='macro')
    recall = recall_score(true_y, pred_y, average='macro')
    f1 = f1_score(true_y, pred_y, average='macro')
    return acc, prec, recall, f1


def plot_history(history: keras.History , output_dir: str = "./outputs/") -> None:
    """与えられた学習履歴からacc, lossの推移をプロットする

    Arguments:
        history {[type]} -- 学習履歴

    Keyword Arguments:
        output_dir {str} -- 出力パス (default: {"./outputs/"})
    """
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc)+1)

    plt.cla()
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Val accuracy')
    plt.legend()
    plt.savefig(output_dir + "acc.png")

    plt.cla()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Val loss')
    plt.legend()
    plt.savefig(output_dir + "loss.png")


def main() -> None:
    """処理を管理する
    """

    # MLflowの実験を開始する（時間計測を開始する）
    mlflow.set_experiment(args.exp_name)
    with mlflow.start_run() as run:

        (train_x, train_y), (val_x, val_y), (test_x, test_y) = get_datasets()
        class_num = train_y.shape[1]
        if args.model == 'fully':
            # 全結合モデルの作成とデータセットを変換する
            reshape_num = train_x.shape[1]*train_x.shape[2]
            train_x = train_x.reshape(-1, reshape_num)
            val_x = val_x.reshape(-1, reshape_num)
            test_x = test_x.reshape(-1, reshape_num)
            model = build_fully_model(reshape_num, class_num)
        elif args.model == 'cnn':
            # CNNモデルの作成とデータセットを変換する
            width, height, ch = train_x.shape[1], train_x.shape[2], 1
            train_x = train_x.reshape(-1, width, height, ch)
            val_x = val_x.reshape(-1, width, height, ch)
            test_x = test_x.reshape(-1, width, height, ch)
            model = build_cnn_model(width, height, ch, class_num)
        else:
            exit(1)

        print("train-> x:{}, y:{}".format(train_x.shape, train_y.shape))
        print("valid-> x:{}, y:{}".format(val_x.shape, val_y.shape))
        print("test -> x:{}, y:{}".format(test_x.shape, test_y.shape))

        # callback設定
        if not os.path.exists("./outputs"):
            os.mkdir("./outputs")
        model_path = "./outputs/best_model.h5"
        csv_path = "./outputs/log.csv"
        cb_csv = CSVLogger(csv_path)
        cb_checkpoint = ModelCheckpoint(
            model_path,
            monitor='val_loss', verbose=0,
            save_best_only=True,
            save_weights_only=True, mode='auto', period=1)

        # 学習を開始
        history = model.fit(train_x, train_y, batch_size=args.b, epochs=args.e,
                            callbacks=[cb_csv, cb_checkpoint],
                            validation_data=[val_x, val_y])
        plot_history(history)

        # 学習済みモデルを読み込む
        if args.model == 'fully':
            model = build_fully_model(reshape_num, class_num, weight_path=model_path)
        elif args.model == 'cnn':
            model = build_cnn_model(width, height, ch, class_num, weight_path=model_path)

        # メトリクスを計算
        val_acc, val_prec, val_recall, val_f1 = calc_metrics(
            model, val_x, val_y)
        test_acc, test_prec, test_recall, test_f1 = calc_metrics(
            model, test_x, test_y)

        # MLflowにパラメータを記録する
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("epoch_num", args.e)
        mlflow.log_param("batch_size", args.b)

        # MLflowにメトリクスを記録する
        mlflow.log_metric("val_acc", val_acc)
        mlflow.log_metric("val_precision", val_prec)
        mlflow.log_metric("val_recall", val_recall)
        mlflow.log_metric("val_f1", val_f1)
        # 辞書でまとめて記録することも可能！（ちなみにパラメータも同様のことができる）
        mlflow.log_metrics({"test_acc": test_acc,
                            "test_presicion": test_prec,
                            "test_recall": test_recall,
                            "test_f1": test_f1})

        # その他に保存したいものを記録する
        mlflow.log_artifact(csv_path)  # ファイルパスを与えてやる
        mlflow.log_artifact("./outputs/acc.png")
        mlflow.log_artifact("./outputs/loss.png")
        # kerasの場合、モデルをmlflow.kerasで保存できる
        mlflow.keras.log_model(model, "models")


if __name__ == "__main__":
    print(args)
    main()
