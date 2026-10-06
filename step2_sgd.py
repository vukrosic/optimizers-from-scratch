"""Step 2. Stochastic gradient descent: a real network, a few pictures at a time.

Looking at all 1,297 pictures before every step is exact but slow. SGD looks at 32 random
pictures, takes a step, and repeats. Each step is a rough guess, but there are 40 times more steps.
"""
import numpy as np

from net import XTR, accuracy, report, steps_to, train


class SGD:
    def __init__(self, lr):
        self.lr = lr

    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g                         # the rule from step 1, for every weight


def run():
    looks = 10 * len(XTR)                            # both get to look at 12,970 pictures in total
    full, full_curve = train(SGD(0.1), steps=10, batch=None, every=1)
    mini, mini_curve = train(SGD(0.1), steps=looks // 32, batch=32, every=1)
    print(f'every picture each step:  10 steps  -> loss {full_curve[-1]:.3f}, test {accuracy(full):.1%}')
    print(f'32 pictures each step:   {looks // 32} steps  -> loss {mini_curve[-1]:.3f}, test {accuracy(mini):.1%}')
    res = report('SGD, learning rate 0.1', lambda: SGD(0.1))            # step 8 finds these learning rates
    return {'full': full_curve, 'mini': mini_curve, 'full_acc': accuracy(full), 'mini_acc': accuracy(mini),
            'mini_steps': looks // 32, 'res': res}


if __name__ == '__main__':
    run()
