"""Step 3. Momentum: keep some of the last step.

In the valley, plain descent bounces from wall to wall and crawls along the floor. Momentum adds up
the recent gradients into a velocity. Across the valley the bounces cancel; along it they add up.
"""
import numpy as np

from net import report
from step1_gradient_descent import descend, verdict, works
from step2_sgd import SGD


class Momentum:
    def __init__(self, lr, beta=0.9):
        self.lr, self.beta, self.v = lr, beta, None

    def step(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self.v):
            v *= self.beta                           # keep 90 % of the old velocity
            v += g                                   # add the new gradient
            p -= self.lr * v


def run():
    plain, fast = descend(SGD(0.019)), descend(Momentum(0.019))
    print(f'valley, learning rate 0.019, plain descent:  {verdict(plain)}')
    print(f'valley, learning rate 0.019, momentum:       {verdict(fast)}')
    print(f'learning rates that reach the bottom, out of 101:  plain {works(SGD)}, momentum {works(Momentum)}')
    a = report('SGD, learning rate 0.1', lambda: SGD(0.1))
    b = report('momentum, learning rate 0.0178', lambda: Momentum(0.0178))
    return {'plain': plain, 'fast': fast, 'works_plain': works(SGD), 'works_fast': works(Momentum), 'sgd': a, 'mom': b}


if __name__ == '__main__':
    run()
