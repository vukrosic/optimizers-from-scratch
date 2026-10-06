"""Step 6. AdamW: weight decay done properly.

Weight decay shrinks every weight a little at each step, so the network cannot lean on a few huge
weights to memorise its training pictures. The old way adds the shrink to the gradient, and Adam
then divides it by the gradient size like everything else. AdamW shrinks the weight directly.
Loshchilov and Hutter, 2017.
"""
import numpy as np

from net import XTR, YTR, accuracy, train
from step5_adam import Adam


class AdamL2(Adam):
    """The old way: add decay * weight to the gradient, then run Adam as usual."""
    def __init__(self, lr, decay):
        super().__init__(lr)
        self.decay = decay

    def step(self, params, grads):
        super().step(params, [g + self.decay * p if p.ndim == 2 else g for p, g in zip(params, grads)])


class AdamW(Adam):
    """The fix: shrink the weight itself, then run Adam on the plain gradient."""
    def __init__(self, lr, decay):
        super().__init__(lr)
        self.decay = decay

    def step(self, params, grads):
        for p in params:
            if p.ndim == 2:                          # matrices only; biases are left alone
                p *= 1 - self.lr * self.decay
        super().step(params, grads)


def shrink(make, sizes=(1.0, 0.1, 0.01), steps=200):
    """Three weights that start at 1. Their gradients only flip sign (+g, -g, +g, ...), so the loss
    does not want them to go anywhere. Only the decay moves them. Where is each after 200 steps?"""
    w, opt = np.ones((1, 3)), make()
    for t in range(steps):
        opt.step([w], [np.array([sizes]) * (1 if t % 2 == 0 else -1)])
    return w[0].tolist()


def noisy(wrong=0.3, n=300, seed=0):
    """A small training set where 3 labels in 10 are replaced by a random digit: something to memorise."""
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(XTR))[:n]
    x, y = XTR[pick], YTR[pick].copy()
    flip = rng.random(n) < wrong
    y[flip] = rng.integers(0, 10, int(flip.sum()))
    return x, y, int((y != YTR[pick]).sum())


def fit(make, seeds=5, steps=2000):
    x, y, _ = noisy()
    runs = [train(make(), steps, seed, x=x, y=y, every=steps)[0] for seed in range(seeds)]
    acc = float(np.mean([accuracy(p) for p in runs]))
    size = float(np.mean([np.sqrt(sum((p ** 2).sum() for p in ps[0::2])) for ps in runs]))
    return acc, size


def run():
    old = shrink(lambda: AdamL2(0.01, 0.1))
    new = shrink(lambda: AdamW(0.01, 0.1))
    print('three weights at 1.0, gradient sizes 1, 0.1, 0.01, decay 0.1, 200 steps:')
    print('  decay inside the gradient:', ' '.join(f'{v:.2f}' for v in old))
    print('  AdamW:                    ', ' '.join(f'{v:.2f}' for v in new))
    out = {'old': old, 'new': new}
    print(f'300 pictures, {noisy()[2]} of them labelled wrong, 2000 steps:')
    for name, make in (('Adam, no decay', lambda: Adam(0.003)), ('old way, decay 0.03', lambda: AdamL2(0.003, 0.03)),
                       ('old way, decay 0.1', lambda: AdamL2(0.003, 0.1)), ('AdamW, decay 1.0', lambda: AdamW(0.003, 1.0))):
        acc, size = fit(make)
        out[name] = (acc, size)
        print(f'  {name:<20} test {acc:.1%}   size of all weights {size:.1f}')
    return out


if __name__ == '__main__':
    run()
