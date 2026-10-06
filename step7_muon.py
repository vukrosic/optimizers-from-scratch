"""Step 7. Muon: treat a layer's weights as one matrix, not as separate numbers.

A gradient matrix is usually lopsided: it pushes hard in one or two directions and barely at all in
the others. Muon keeps the directions but gives them all the same strength (it "orthogonalizes"
the update), using only matrix multiplications. Keller Jordan and others, 2024.
"""
import numpy as np

from net import XTR, YTR, loss_and_grads, new_weights, report
from step5_adam import Adam


def orthogonalize(G, rounds=5):
    """Newton-Schulz: push every singular value of G toward 1, with matrix products only."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (np.linalg.norm(G) + 1e-7)               # now every strength is at most 1
    wide = X.shape[0] > X.shape[1]
    if wide:
        X = X.T
    for _ in range(rounds):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X          # small strengths grow fast, strengths near 1 stay
    return X.T if wide else X


class Muon:
    def __init__(self, lr, beta=0.95):
        self.lr, self.beta, self.m = lr, beta, None
        self.adam = Adam(0.01)                       # biases are not matrices: Adam keeps those

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
        for p, g, m in zip(params, grads, self.m):
            if p.ndim == 2:
                m *= self.beta
                m += g                               # momentum, as in step 3
                update = orthogonalize(g + self.beta * m)
                p -= self.lr * max(1, p.shape[1] / p.shape[0]) ** 0.5 * update
        biases = [i for i, p in enumerate(params) if p.ndim == 1]
        self.adam.step([params[i] for i in biases], [grads[i] for i in biases])


def strengths():
    """The singular values of one real gradient matrix (layer 3, 32 x 32), before and after."""
    _, grads = loss_and_grads(new_weights(), XTR[:32], YTR[:32])
    before = np.linalg.svd(grads[4], compute_uv=False)
    after = np.linalg.svd(orthogonalize(grads[4]), compute_uv=False)
    rounds = [np.linalg.svd(orthogonalize(grads[4], r), compute_uv=False) for r in range(6)]
    return before, after, rounds


def run():
    before, after, rounds = strengths()
    top = before / before[0]
    print('strength of each direction in one gradient matrix (strongest = 1):')
    print('  before:', ' '.join(f'{s:.2f}' for s in top[:8]), '...', f'{top[-1]:.5f}')
    print('  after: ', ' '.join(f'{s:.2f}' for s in after[:8]), '...', f'{after[-1]:.2f}')
    print(f'  directions above a tenth of the strongest: before {int((top > 0.1).sum())} of {len(top)}, '
          f'after {int((after > 0.1 * after.max()).sum())} of {len(after)}')
    a = report('Adam, learning rate 0.01', lambda: Adam(0.01))
    b = report('Muon, learning rate 0.00562', lambda: Muon(0.00562))
    return {'before': top, 'after': after, 'rounds': rounds, 'adam': a, 'muon': b}


if __name__ == '__main__':
    run()
