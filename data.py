"""The data for every step: 1,797 handwritten digits, each 8 x 8 pixels.

From the UCI "Optical Recognition of Handwritten Digits" set (Alpaydin and Kaynak, 1998),
the copy that ships with scikit-learn. Pixel values 0..16 are scaled to 0..1.
"""
import gzip
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(n_test=500, seed=0):
    raw = np.loadtxt(gzip.open(os.path.join(HERE, 'digits.csv.gz'), 'rt'), delimiter=',')
    x, y = raw[:, :64] / 16.0, raw[:, 64].astype(int)
    order = np.random.default_rng(seed).permutation(len(x))
    x, y = x[order], y[order]
    return x[n_test:], y[n_test:], x[:n_test], y[:n_test]      # train x, train y, test x, test y


def canvas(x, dx=0, dy=0, size=12):
    """Put each 8 x 8 digit on an empty size x size page, moved dx pixels right and dy pixels down
    from the centre. Returns shape (n, 1, size, size): one channel, like a grey photograph."""
    x = np.asarray(x).reshape(-1, 8, 8)
    out, m = np.zeros((len(x), 1, size, size)), (size - 8) // 2
    out[:, 0, m + dy:m + dy + 8, m + dx:m + dx + 8] = x
    return out


def show(img, levels=' .:*#'):
    """Print one picture with text characters."""
    img = np.asarray(img)
    side = int(round(np.sqrt(img.size)))
    for row in img.reshape(side, side):
        print(''.join(levels[min(len(levels) - 1, int(max(v, 0) * len(levels)))] * 2 for v in row))


if __name__ == '__main__':
    xtr, ytr, xte, yte = load()
    print(len(xtr), 'training pictures,', len(xte), 'test pictures, 64 pixels each')
    print('this one is a', ytr[0])
    show(xtr[0])
