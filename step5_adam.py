"""Step 5. Adam: momentum and RMSProp together, plus a fix for the first steps.

m is the running average of the gradient (momentum). v is the running average of its square
(RMSProp). Both start at zero, so early on they are far too small; dividing by (1 - beta^t)
corrects that. Kingma and Ba, 2014.
"""
import numpy as np

from net import report
from step1_gradient_descent import descend, verdict, works
from step2_sgd import SGD
from step3_momentum import Momentum
from step4_rmsprop import RMSProp


class Adam:
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8, fix=True):
        self.lr, self.b1, self.b2, self.eps, self.fix = lr, b1, b2, eps, fix
        self.m, self.v, self.t = None, None, 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g                   # average gradient: the direction
            v *= self.b2
            v += (1 - self.b2) * g * g               # average squared gradient: the scale
            m_hat = m / (1 - self.b1 ** self.t) if self.fix else m
            v_hat = v / (1 - self.b2 ** self.t) if self.fix else v
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def first_steps(fix, grads=(5.0, 5.0, 5.0, 5.0, 5.0)):
    """One weight, the same gradient five times, learning rate 1: how far does each step move it?"""
    w, opt, moves = np.array([0.0]), Adam(1.0, fix=fix), []
    for g in grads:
        before = w.copy()
        opt.step([w], [np.array([g])])
        moves.append(float(before[0] - w[0]))
    return moves


def run():
    path = descend(Adam(0.3))
    print(f'valley, Adam, learning rate 0.3:  {verdict(path)}')
    count = {name: int(works(make)) for name, make in (('plain', SGD), ('momentum', Momentum), ('RMSProp', RMSProp), ('Adam', Adam))}
    print('learning rates that reach the bottom, out of 101: ', ',  '.join(f'{k} {v}' for k, v in count.items()))
    raw, fixed = first_steps(False), first_steps(True)
    print('first five steps without the fix:', ' '.join(f'{x:.2f}' for x in raw))
    print('first five steps with the fix:   ', ' '.join(f'{x:.2f}' for x in fixed))
    a = report('RMSProp, learning rate 0.00316', lambda: RMSProp(0.00316))
    b = report('Adam, learning rate 0.01', lambda: Adam(0.01))
    return {'path': path, 'works': count, 'raw': raw, 'fixed': fixed, 'rms': a, 'adam': b}


if __name__ == '__main__':
    run()
