import os
import sys
import glob
import re
import yaml
import datetime
import shutil

import numpy as np
import chainer
from chainer.backends import cuda
from chainer.optimizers import Adam
from chainer.datasets import TupleDataset
from chainer.iterators import SerialIterator
from chainer.training import Trainer
from chainer.training import extensions
from chainer.serializers import load_npz

from predicting.model import Estimator
from predicting.updater import EstimatorUpdater
from predicting.validator import Validator

# 設定ファイル
CONFIG_FILE = "config.yaml"
# 学習済みモデルの出力先
MODEL_DIR = "result"
# 検証ログの出力先
VALIDATE_DIR = "validate"

def setup():
    # 設定ファイルの読み込み
    with open(CONFIG_FILE, "r") as f:
        config = yaml.load(f)

    xp = np if not config["use_gpu"] else cuda.cupy
    
    # 学習結果出力先の設定
    restart = config["restart_dir"] is not None
    if restart:
        result_children_dir = config["restart_dir"]
    else:
        result_children_dir = "result_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
    result_dir = os.path.join(config["result_dir"], result_children_dir)
    result_dir_train = os.path.join(result_dir, MODEL_DIR)
    result_dir_val = os.path.join(result_dir, VALIDATE_DIR)

    # 学習データの読み込み
    train_scores = []
    with open(os.path.join(config["score_dir"], config["train_list"]), "r") as tr_f:
        train_info = list(map(lambda x: x.split("\n")[0], tr_f.readlines()))
        train_paths = list(map(lambda x: os.path.join(config["score_dir"], x.split("\t")[0]), train_info))
        train_score_lvs = list(map(lambda x: int(x.split("\t")[1])-1, train_info))

    for idx, npy_path in enumerate(train_paths):
        score = xp.load(npy_path)
        score[:, 8] /= 100.0
        # 譜面を小節ごとに区切る
        score = score.reshape((-1, 1728))
        train_scores.append(score)
        sys.stdout.write("\rtrain score loaded: {0:4d}/{1}".format(idx+1, len(train_paths)))
    sys.stdout.write("\n")

    # 検証データの読み込み
    val_scores = []
    val_score_names = []
    with open(os.path.join(config["score_dir"], config["validate_list"]), "r") as val_f:
        val_info = list(map(lambda x: x.split("\n")[0], val_f.readlines()))
        val_paths = list(map(lambda x: os.path.join(config["score_dir"], x.split("\t")[0]), val_info))
        val_score_lvs = list(map(lambda x: int(x.split("\t")[1])-1, val_info))

    for idx, npy_path in enumerate(val_paths):
        score = xp.load(npy_path)
        score[:, 8] /= 100.0
        # 譜面を小節ごとに区切る
        score = score.reshape((-1, 1728))
        val_scores.append(score)
        score_name = os.path.basename(npy_path)
        val_score_names.append(score_name)
        sys.stdout.write("\rvalidate score loaded: {0:4d}/{1}".format(idx+1, len(val_paths)))
    sys.stdout.write("\n")

    # optimizer
    model = Estimator()
    if xp is not np:
        model.to_device("@cupy:0")    
    optimizer = Adam(float(config["lr"]))
    optimizer.setup(model)

    # iterator, updater, trainer, extension
    train_dataset = TupleDataset(train_scores, train_score_lvs)
    train_iterator = SerialIterator(train_dataset, int(config["batch_size"]))
    val_dataset = TupleDataset(val_scores, val_score_lvs, val_score_names)
    val_iterator = SerialIterator(val_dataset, int(config["batch_size"]), repeat=False, shuffle=False)

    updater = EstimatorUpdater(iterator=train_iterator, optimizer=optimizer)
    trainer = Trainer(updater, stop_trigger=(config["epochs"], "epoch"), out=result_dir_train)

    trainer.extend(Validator(val_iterator, result_dir_val), trigger=(1, "epoch"))
    trainer.extend(extensions.snapshot(filename="snapshot_epoch_{.updater.epoch}"))
    trainer.extend(extensions.LogReport(trigger=(1, "epoch")), trigger=(1, "epoch"))
    trainer.extend(extensions.PrintReport(["epoch", "train/loss", "train/acc", "val/loss", "val/acc", "val/rough_acc"]))
    trainer.extend(extensions.ProgressBar(update_interval=5))

    if restart:
        # 学習を再開するモデルを特定
        snapshot_path_format = os.path.join(result_dir_train, "snapshot_epoch_*")
        snapshots = [os.path.basename(fname) for fname in glob.glob(snapshot_path_format)]
        if len(snapshots) == 0:
            print("There does not exist a model to restart training.")
            exit()
        else:
            pattern = re.compile("snapshot_epoch_([0-9]+)")
            snapshot_epochs = list(map(lambda x: int(pattern.search(x).group(1)), snapshots))
            prev_snapshot_idx = snapshot_epochs.index(max(snapshot_epochs))
            prev_snapshot = snapshots[prev_snapshot_idx]
            
        load_npz(os.path.join(result_dir_train, prev_snapshot), trainer)

    shutil.copy2(CONFIG_FILE, result_dir)

    return trainer

if __name__ == "__main__":
    trainer = setup()
    trainer.run()
