import os
import yaml

import numpy as np
import chainer
import chainer.functions as F
from chainer.backends import cuda
from chainer.datasets import TupleDataset
from chainer.iterators import SerialIterator
from chainer.serializers import load_npz
from sklearn.decomposition import PCA
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

from lib.model import Estimator

# 学習済みモデルの配置先
MODEL_DIR = "result"

def get_colormap():
    """
    [1]red _ yellow _ green _ lightblue _ blue _ purple[12]
    (255,0,0)_(255,255,0)_(0,255,0)_(0,255,255)_(0,0,255)_(255,0,255)
    R: 255_ 255_   0_   0_   0_ 255
    G:   0_ 255_ 255_ 255_   0_   0
    B:   0_   0_   0_ 255_ 255_ 255
    """
    # lv        1        2        3            4        5        6            7            8            9       10       11   12
    # r_list = [1.0,     1.0,     1.0, 1.0-0.8/2.2,     0.0,     0.0,         0.0,         0.0,         0.0, 0.2/2.2, 1.2/2.2, 1.0]
    # g_list = [0.0, 1.0/2.2, 2.0/2.2,         1.0,     1.0,     1.0, 1.0-0.4/2.2, 1.0-0.9/2.2, 1.0-1.4/2.2,     0.0,     0.0, 0.0]
    # b_list = [0.0,     0.0,     0.0,         0.0, 0.2/2.2, 1.6/2.2,         1.0,         1.0,         1.0,     1.0,     1.0, 1.0]
    r_list = [1.0,     1.0,     1.0,         1.0, 1.0-0.8/2.2,     0.0,     0.0,         0.0,         0.0, 0.2/2.2, 1.2/2.2, 1.0]
    g_list = [0.0, 0.6/2.2, 1.5/2.2,     2.0/2.2,         1.0,     1.0,     1.0, 1.0-0.4/2.2, 1.0-1.2/2.2,     0.0,     0.0, 0.0]
    b_list = [0.0,     0.0,     0.0,         0.0,         0.0, 0.2/2.2, 1.6/2.2,         1.0,         1.0,     1.0,     1.0, 1.0]

    cmap = [(r, g, b) for r, g, b in zip(r_list, g_list, b_list)]

    return cmap

def concat_batch(batch: list):
    scores = []
    names = []
    for (score, name) in batch:
        scores.append(score)
        names.append(name)

    return scores, names

if __name__ == "__main__":
    # 設定ファイルの読み込み
    with open("config.yaml", "r") as f:
        config = yaml.load(f)

    xp = np if not config["use_gpu"] else cuda.cupy

    # 検証データの読み込み
    test_scores = []
    test_score_names = []
    with open(os.path.join(config["score_dir"], config["test_list"]), "r") as test_f:
        test_paths = list(map(lambda x: os.path.join(config["score_dir"], x.split("\n")[0]), test_f.readlines()))
    for idx, npy_path in enumerate(test_paths):
        score = xp.load(npy_path)
        score[:, 8] /= 100.0
        # 譜面を流れてくる順に直して小節ごとに区切る
        score = score[::-1].reshape((-1, 1728))
        test_scores.append(score)
        score_name = os.path.basename(npy_path)
        test_score_names.append(score_name)

    # 推論の準備
    chainer.global_config.train = False

    model = Estimator()
    model.to_gpu()
    test_model_name = "snapshot_epoch_" + str(config["test_model_epoch"])
    test_model_path = os.path.join(config["result_dir"], config["test_dir"], MODEL_DIR, test_model_name)
    load_npz(test_model_path, model, path="updater/model:main/")

    test_dataset = TupleDataset(test_scores, test_score_names)
    test_iterator = SerialIterator(test_dataset, int(config["batch_size"]), repeat=False, shuffle=False)
    
    test_log_name = "test_epoch_" + str(config["test_model_epoch"]) + ".txt"
    test_log_path = os.path.join(config["result_dir"], config["test_dir"], test_log_name)
    unit_list = []
    est_lv_list = []    

    # 推論
    with open(test_log_path, "w") as log_f:
        log_f.write("\t".join(["score_name", "lv", "likelihoods"]) + "\n")
        remaining = len(test_dataset)
        while remaining > 0:
            batch_size = min(test_iterator.batch_size, remaining)
            scores, names = concat_batch(test_iterator.next())
            est = model(scores[:batch_size], True)
            unit_list.extend(model._h.data.get())

            est_softmax = F.softmax(est)
            est_argmax = F.argmax(est_softmax, axis=1)
            est_lv_list.extend(est_argmax.data.get())

        # 尤度をログに出力
            for element, lv, name in zip(est_softmax, est_argmax, names):
                log = [os.path.splitext(name)[0], str(int(lv.data)+1)]
                log.extend(["{:.3f}".format(float(value.data)) for value in element])
                log_f.write("\t".join(log) + "\n")

            remaining -= batch_size

    # 主成分分析
    if config["show_pca"]:
        pca = PCA(n_components=3)
        units = np.array(unit_list)
        pca.fit(units)
        features = pca.fit_transform(units)
        print("explained_variance: {}".format(pca.explained_variance_ratio_))

        # 散布図準備
        fig = pyplot.figure()
        ax = Axes3D(fig)
        ax.set_title("principal component of hidden units")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")

        # 点の色
        colors = np.zeros((units.shape[0], 3))
        cmap = get_colormap()
        for lv in range(12):
            lv_idx = np.array(est_lv_list) == lv
            colors[lv_idx] = cmap[lv]

        # プロット
        scat = ax.scatter3D(features[:, 0], features[:, 1], zs=features[:, 2], c=colors)

        # マウスオーバーで譜面名が出るように設定
        anno = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points", bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
        anno.set_visible(False)

        def update_anno(ind):
            idx = ind["ind"][0]
            coord = scat.get_offsets()[idx]
            anno.xy = coord
            text = test_score_names[idx]
            anno.set_text(text)
        
        def hover(event):
            if event.inaxes == ax:
                content, ind = scat.contains(event)
                if content:
                    update_anno(ind)
                    anno.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if anno.get_visible():
                        anno.set_visible(False)
                        fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)

        # pyplot.legend()
        pyplot.show()
        pyplot.clf()