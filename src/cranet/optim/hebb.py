import cranet

from .learning_rule import LearningRule


class Hebb(LearningRule):
    def __init__(self, c=0.1):
        super().__init__()
        self.c = c

    def update(self, inputs, w):
        d_ws = cranet.zeros(inputs.size(0))
        for idx, x in enumerate(inputs):
            y = cranet.dot(w, x)

            d_w = cranet.zeros(w.shape)
            for i in range(y.shape[0]):
                for j in range(x.shape[0]):
                    d_w[i, j] = self.c * x[j] * y[i]

            d_ws[idx] = d_w

        return cranet.mean(d_ws, dim=0)
