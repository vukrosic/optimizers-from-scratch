"""Step 4. RMSProp: a separate step size for every weight.

One learning rate has to suit the steepest weight, so the flat ones barely move. RMSProp remembers
how big each weight's gradient usually is and divides by it: steep weights take smaller steps,
flat weights take bigger ones.
"""
import numpy as np

from net import XTR, YTR, loss_and_grads, new_weights, report
from step1_gradient_descent import descend, loss, verdict
from step3_momentum import Momentum


class RMSProp:
    def __init__(self, lr, rho=0.9, eps=1e-8):
        self.lr, self.rho, self.eps, self.s = lr, rho, eps, None

    def step(self, params, grads):
        if self.s is None:
            self.s = [np.zeros_like(p) for p in params]
        for p, g, s in zip(params, grads, self.s):
            s *= self.rho
            s += (1 - self.rho) * g * g              # running average of the squared gradient
            p -= self.lr * g / (np.sqrt(s) + self.eps)


def spread():
    """How different are the gradients inside the real network? Typical size, layer by layer."""
    _, grads = loss_and_grads(new_weights(), XTR, YTR)
    sizes = [float(np.abs(g).mean()) for g in grads[0::2]]
    every = np.abs(np.concatenate([g.ravel() for g in grads]))
    every = every[every > 0]
    return sizes, float(np.percentile(every, 99) / np.percentile(every, 1))


def run():
    path = descend(RMSProp(0.1))
    print('valley, RMSProp, learning rate 0.1, first steps:', '  '.join(f'({a:.2f}, {b:.2f})' for a, b in path[:4]))
    print(f'  loss after 100 steps {loss(path[100]):.2f}, after 1000 steps {loss(path[-1]):.2f}: close, but it never settles')
    sizes, ratio = spread()
    print('typical gradient size in each layer of the network:', ' '.join(f'{s:.4f}' for s in sizes))
    print(f'a large gradient (top 1 %) is {ratio:,.0f} times a small one (bottom 1 %)')
    a = report('momentum, learning rate 0.0178', lambda: Momentum(0.0178))
    b = report('RMSProp, learning rate 0.00316', lambda: RMSProp(0.00316))
    return {'path': path, 'sizes': sizes, 'ratio': ratio, 'mom': a, 'rms': b}


if __name__ == '__main__':
    run()
