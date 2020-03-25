from chainer import Chain
import chainer.links as L
import chainer.functions as F

class Estimator(Chain):
    def __init__(self):
        super(Estimator, self).__init__()
        with self.init_scope():
            self.layer1 = L.NStepLSTM(2, 1728, 256, 0.3)
            self.layer2 = L.Linear(256, 12)
    
    def __call__(self, x, validate=False):
        _, _, h = self.layer1(None, None, x)
        h = [t[-1] for t in h]
        # 後でPCAやるために隠れ層を保持できるようにした
        self._h = F.stack(h)

        return self.layer2(self._h)