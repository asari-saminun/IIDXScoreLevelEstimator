# IIDXScoreLevelEstimator

IIDXScoreEstimatorは、beatmaniaIIDXで再現可能な譜面のレベルを推論するツールです。

## 動作環境

最低限必要なもの

- Python 3.6
- Numpy
- Chainer 7.2.0 (v6以上なら動くと思う)

※GPU利用の場合
- Cuda 9以上
- Cupy 7.2.0 (Chainerと同じバージョン)

※学習曲線や主成分分析結果を出力したい場合
- matplotlib 2以上
- scikit-learn

## レベル推論方法

以下の手順で行います。

1. 譜面データを用意する。
2. estimating/config.yamlを編集し、各種設定を行う。
3. score_test.txtに譜面データのパスを書き込む。
4. estimating/test.pyを実行する。
5. 推論結果を確認する。

### 1. 譜面データの用意

譜面はnumpy.ndarrayの2次元配列として入力します。

配列の1次元目はノーツのタイミングを表します。タイミングの最小単位は4分音符の48分の1（192分音符相当）です。
2次元目は譜面のレーン（ターンテーブル+1～7鍵）とBPMを表します。　配列のBPM部分を除く各要素の値を以下のように設定してください。

|  | ノーツ無し | ノーツあり | CN・HCN・BSS始点/終点 | CN・HCN・BSS途中 |
|:-:|:-:|:-:|:-:|:-:|
| 値 | 0 | 1 | 1 | 0.5 |

※CN=チャージノート、HCN=ヘルチャージノート、BSS=バックスピンスクラッチ

上記の要領で作成した全小節をひと繋ぎにし、1次元目の長さが192の倍数になるよう調節してください。

### 2. config.yamlの編集

実行前にestimate/config.yamlを編集します。とりあえず推論したいだけであれば、編集する必要のある項目はscore_dirとuse_gpuのみです。

#### 全体設定

- score_dir: 譜面データ(npy形式)の格納ディレクトリ。
- result_dir: 学習/テスト結果の出力先ディレクトリ。
- use_gpu: Trueならば学習/推論時にGPUを利用する。

#### 学習用の設定

- train_list: 学習データのパスを書き込んだファイル名。score_dirの下に配置してください。
- validate_list: 検証データのパスを書き込んだファイル名。score_dirの下に配置してください。
- epochs: 学習回数。
- batch_size: バッチサイズ。
- lr: 学習率。
- restart_dir: 学習を再開する学習結果ディレクトリ。

#### 推論用の設定

- test_list: テストデータのパスを書き込んだファイル。score_dirの下に配置してください。
- test_dir: 推論に使用する学習済みモデルの格納されたディレクトリ。
- test_model_epoch: 学習モデルの学習回数。
- show_pca: Trueならばtest.py実行時にテストデータの主成分分析を行う。

### 3. score_test.txtの編集

config.yamlのscore_dir直下にscore_test.txt（config.yamlのtest_listに指定したファイル名）を作成し、1.で作成した譜面データファイルのパスを書き込みます。ファイルのパスはscore_dirからの相対パスとし、1行に1つずつ羅列してください。



### 4. 推論の実行

カレントディレクトリをestimatingディレクトリにし、test.pyを実行してください。推論結果はデフォルトではresults/result_sample/result/test_epoch_48.txtに出力されます。正確には (result_dir)/(test_dir)/result/test_model_(test_model_epoch).txtが出力ファイルパスです。ただし括弧内の値はconfig.yamlの内容が反映されます。

### 5. 推論結果の確認

4.で出力されるファイルには以下の内容が含まれます。
