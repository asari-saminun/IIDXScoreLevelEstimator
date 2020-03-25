import os

from chainer import reporter
from chainer.backends import cuda
from chainer.iterators import SerialIterator
from chainer.training import Trainer
import chainer.functions as F

def concat_batch(batch: list):
    scores = []
    levels = []
    names = []
    for (score, level, name) in batch:
        scores.append(score)
        levels.append(level)
        names.append(name)

    return scores, levels, names

def rough_accuracy(est_argmax, lv_array):
    diff = est_argmax - lv_array
    diff = F.absolute(diff.data.astype(float))
    acc = F.sum((diff.data <= 1).astype(float))

    return acc

class Validator:

    def __init__(self, iterator: SerialIterator, log_dir: str):
        self.iterator = iterator
        self.log_dir = log_dir
        self.log_name = os.path.join(log_dir, "validate_epoch{}.log")

        os.makedirs(log_dir, exist_ok=True)
        self.name = "val"

    def __call__(self, trainer: Trainer):
        self.iterator.reset()
        opt = trainer.updater.get_optimizer("main")
        log_path = self.log_name.format(trainer.updater.epoch)
        loss_total = 0.0
        acc_total = 0.0
        rough_acc_total = 0.0

        with open(log_path, "w") as log_f:
            log_f.write("\t".join(["score_name", "lv", "likelihoods"]) + "\n")
            remaining = len(self.iterator.dataset)
            while remaining > 0:
                batch_size = min(self.iterator.batch_size, remaining)
                scores, levels, names = concat_batch(self.iterator.next())
                est = opt.target(scores[:batch_size], True)

                lv_array = opt.target.xp.array(levels, dtype=int)
                est_softmax = F.softmax(est)
                est_argmax = F.argmax(est_softmax, axis=1)
                loss_total += F.softmax_cross_entropy(est, lv_array).data * batch_size
                acc_total += F.accuracy(est, lv_array).data * batch_size
                rough_acc_total += rough_accuracy(est_argmax, lv_array)

                # 尤度をログに出力
                for element, act_lv, name in zip(est_softmax, est_argmax, names):
                    log = [os.path.splitext(name)[0], str(int(act_lv.data)+1)]
                    log.extend(["{:.3f}".format(float(value.data)) for value in element])
                    log_f.write("\t".join(log) + "\n")

                remaining -= batch_size

        reporter.report({
            "val/loss": loss_total / len(self.iterator.dataset),
            "val/acc": acc_total / len(self.iterator.dataset),
            "val/rough_acc": rough_acc_total / len(self.iterator.dataset)
        })