from chainer import reporter
from chainer.training import StandardUpdater
import chainer.functions as F

def concat_batch(batch: list):
    scores = []
    levels = []
    for (score, level) in batch:
        scores.append(score)
        levels.append(level)

    return scores, levels

class EstimatorUpdater(StandardUpdater):
    def __init__(self, **kwargs):
        super(EstimatorUpdater, self).__init__(**kwargs)

    def update_core(self):
        # イテレータには譜面データと正解のレベルを格納している
        scores, levels = concat_batch(self.get_iterator("main").next())
        opt = self.get_optimizer("main")

        est = opt.target(scores)
        lv = opt.target.xp.array(levels, dtype=int)
        loss = F.softmax_cross_entropy(est, lv)

        reporter.report({
            "train/loss": loss,
            "train/acc": F.accuracy(est, lv)
        })

        opt.target.cleargrads()
        loss.backward()
        loss.unchain_backward()
        opt.update()