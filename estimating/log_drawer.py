import os
import json
import yaml
import math

import numpy as np
from matplotlib import pyplot

# 設定ファイル
CONFIG_FILE = "config.yaml"
# 学習済みモデルの格納先
MODEL_DIR = "result"

def parse_log(log_dir, log_info):
    # ログの解析
    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    val_rough_acc = []

    for idx in range(len(log_info)):
        epochs.append(log_info[idx]["epoch"])
        train_loss.append(log_info[idx]["train/loss"])
        train_acc.append(log_info[idx]["train/acc"])
        val_loss.append(log_info[idx]["val/loss"])
        val_acc.append(log_info[idx]["val/acc"])
        val_rough_acc.append(log_info[idx]["val/rough_acc"])

    # lossやaccの最大最小を表示
    print("[train/loss] min={0:.6f} ({1:3d}epoch)".format(min(train_loss), train_loss.index(min(train_loss))+1))
    print("[train/ acc] max={0:.6f} ({1:3d}epoch)".format(max(train_acc), train_acc.index(max(train_acc))+1))
    print("[val  /loss] min={0:.6f} ({1:3d}epoch)".format(min(val_loss), val_loss.index(min(val_loss))+1))
    print("[val  / acc] max={0:.6f} ({1:3d}epoch)".format(max(val_acc), val_acc.index(max(val_acc))+1))
    print("[val  /racc] max={0:.6f} ({1:3d}epoch)".format(max(val_rough_acc), val_rough_acc.index(max(val_rough_acc))+1))

    epochs_arr = np.array(epochs, dtype=np.uint16)
    train_loss_arr = np.array(train_loss, dtype=np.float32)
    train_acc_arr = np.array(train_acc, dtype=np.float32)
    val_loss_arr = np.array(val_loss, dtype=np.float32)
    val_acc_arr = np.array(val_acc, dtype=np.float32)
    val_rough_acc_arr = np.array(val_rough_acc, dtype=np.float32)    
    
    # 学習ログのグラフ化
    t_fig, t_axis1 = pyplot.subplots()
    t_fig.suptitle("train")
    t_axis2 = t_axis1.twinx()
    t_axis2.spines["left"].set_color("blue")
    t_axis2.spines["right"].set_color("orange")
    t_axis1.plot(epochs_arr, train_loss_arr, label="loss", color="blue")
    t_axis2.plot(epochs_arr, train_acc_arr, label="acc", color="orange")
    t_axis1.set_xlabel("epochs")
    t_axis1.set_ylabel("loss")
    t_axis2.set_ylabel("acc")
    t_axis1.set_ylim([0, math.ceil(max(train_loss + val_loss))])
    t_axis2.set_ylim([0, 1])
    t_handler1, t_label1 = t_axis1.get_legend_handles_labels()
    t_handler2, t_label2 = t_axis2.get_legend_handles_labels()
    t_axis1.legend(t_handler1 + t_handler2, t_label1 + t_label2)
    t_fig.tight_layout()

    t_out_path = os.path.join(log_dir, "train")
    t_fig.savefig(t_out_path)

    # # 検証ログのグラフ化
    v_fig, v_axis1 = pyplot.subplots()
    v_fig.suptitle("validate")
    v_axis2 = v_axis1.twinx()
    v_axis2.spines["left"].set_color("blue")
    v_axis2.spines["right"].set_color("orange")
    v_axis1.plot(epochs_arr, val_loss_arr, label="loss", color="blue")
    v_axis2.plot(epochs_arr, val_acc_arr, label="acc", color="orange")
    v_axis2.plot(epochs_arr, val_rough_acc_arr, label="rough_acc", color="red")
    v_axis1.set_xlabel("epochs")
    v_axis1.set_ylabel("loss")
    v_axis2.set_ylabel("acc")
    v_axis1.set_ylim([0, math.ceil(max(train_loss + val_loss))])
    v_axis2.set_ylim([0, 1])
    v_handler1, v_label1 = v_axis1.get_legend_handles_labels()
    v_handler2, v_label2 = v_axis2.get_legend_handles_labels()
    v_axis1.legend(v_handler1 + v_handler2, v_label1 + v_label2)
    v_fig.tight_layout()

    v_out_path = os.path.join(log_dir, "validate")
    v_fig.savefig(v_out_path)

if __name__ == "__main__":  
    # 設定ファイルの読み込み
    with open(CONFIG_FILE, "r") as f:
        config = yaml.load(f)

    log_dir = os.path.join(config["result_dir"], config["test_dir"])
    log_path = os.path.join(log_dir, MODEL_DIR, "log")
    log_info = json.load(open(log_path, "r"))

    parse_log(log_dir, log_info)  