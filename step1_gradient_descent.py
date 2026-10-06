"""Step 1. Gradient descent: walk downhill.

A loss tells us how wrong the weights are. The gradient says which way is uphill for each weight.
Gradient descent takes a small step the other way, again and again. Here the "network" has two
weights and the loss is a long narrow valley: steep across, almost flat along.
"""
import numpy as np

START = (-10.0, 1.0)


def loss(w):
    return 0.5 * w[0] ** 2 + 50 * w[1] ** 2          # lowest point: both weights 0


def grad(w):
    return np.array([w[0], 100 * w[1]])              # slope along the valley, slope across it


def nudge(w, i, eps=1e-6):
    """The gradient without calculus: move one weight a tiny bit and see how the loss changes."""
    up = np.array(w, dtype=float)
    up[i] += eps
    return (loss(up) - loss(w)) / eps


def gradient_descent(lr, steps=1000):
    w = np.array(START)
    path = [w.copy()]
    for _ in range(steps):
        w = w - lr * grad(w)                         # the whole algorithm
        path.append(w.copy())
        if loss(w) < 0.01 or loss(w) > 1e6:
            break
    return np.array(path)


def descend(opt, steps=1000):
    """The same walk with any optimizer object (steps 2 to 5 use this)."""
    w = np.array(START)
    path = [w.copy()]
    for _ in range(steps):
        opt.step([w], [grad(w)])
        path.append(w.copy())
        if loss(w) < 0.01 or loss(w) > 1e6:
            break
    return np.array(path)


def verdict(path):
    end = loss(path[-1])
    return f'reached the bottom in {len(path) - 1} steps' if end < 0.01 else ('blew up' if end > 1e6 else f'still at loss {end:.2f}')


def works(make, rates=np.geomspace(0.0001, 10, 101)):
    """Out of 101 learning rates from 0.0001 to 10, how many reach the bottom within 1000 steps?"""
    return sum(loss(descend(make(lr))[-1]) < 0.01 for lr in rates)


def run():
    w = np.array(START)
    print('start', START, ' loss', loss(w))
    print('gradient by nudging:', round(nudge(w, 0), 3), round(nudge(w, 1), 3), '  by formula:', grad(w))
    out = {}
    for lr in (0.001, 0.019, 0.021):
        path = gradient_descent(lr)
        out[lr] = path
        print(f'learning rate {lr:<6} {verdict(path)}')
    return out


if __name__ == '__main__':
    run()
