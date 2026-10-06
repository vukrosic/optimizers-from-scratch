"""The network every step trains, and the loop that trains it with any optimizer.

64 pixels in, five hidden layers of 32 neurons, 10 scores out (one per digit): 6,634 weights.
An optimizer is any object with step(params, grads): it changes the weights in place.
"""
import numpy as np

from data import load

np.seterr(divide='ignore', over='ignore', invalid='ignore')   # numpy 2.0 on Apple chips prints false matmul warnings
SIZES = [64, 32, 32, 32, 32, 32, 10]
XTR, YTR, XTE, YTE = load()


def new_weights(seed=0, sizes=SIZES):
    """One matrix and one bias vector per layer, as a flat list: [W1, b1, W2, b2, ...]."""
    rng = np.random.default_rng(seed)
    params = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        params += [rng.normal(0, np.sqrt(2.0 / a), (a, b)), np.zeros(b)]
    return params


def forward(params, x):
    acts = [x]
    for i in range(0, len(params), 2):
        z = acts[-1] @ params[i] + params[i + 1]
        acts.append(np.maximum(z, 0) if i < len(params) - 2 else z)     # ReLU, except the last layer
    return acts


def loss_and_grads(params, x, y):
    """The loss on these pictures and, for every weight, its gradient: which way is uphill and how steep."""
    acts = forward(params, x)
    z = acts[-1] - acts[-1].max(axis=1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)                # softmax: scores -> chances
    n = len(x)
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
    d = p
    d[np.arange(n), y] -= 1
    d /= n
    grads = [None] * len(params)
    for i in range(len(params) - 2, -1, -2):                            # backpropagation, last layer first
        grads[i], grads[i + 1] = acts[i // 2].T @ d, d.sum(axis=0)
        if i:
            d = (d @ params[i].T) * (acts[i // 2] > 0)
    return (loss if np.isfinite(loss) else 99.0), grads


def accuracy(params, x=XTE, y=YTE):
    return float((forward(params, x)[-1].argmax(axis=1) == y).mean())


def train(opt, steps=400, seed=0, batch=32, x=XTR, y=YTR, every=10):
    """Train a fresh network. Returns its weights and the loss on all pictures, measured every 10 steps."""
    params = new_weights(seed)
    rng = np.random.default_rng(seed + 1000)
    curve = []
    for step in range(1, steps + 1):
        pick = rng.integers(0, len(x), batch) if batch else slice(None)   # batch=None: every picture
        _, grads = loss_and_grads(params, x[pick], y[pick])
        opt.step(params, grads)
        if step % every == 0:
            curve.append(loss_and_grads(params, x, y)[0])
    return params, curve


def steps_to(curve, target=0.1, every=10):
    """The first measured step where the loss is under the target, or None."""
    for i, loss in enumerate(curve):
        if loss < target:
            return (i + 1) * every
    return None


def typical(curves):
    """The middle run at every measurement: one unlucky start cannot bend this curve."""
    return np.median(np.array(curves), axis=0)


def report(name, make, seeds=5, steps=400):
    """Train from five different starting weights and print the typical result."""
    runs = [train(make(), steps, seed) for seed in range(seeds)]
    curve = typical([c for _, c in runs])
    acc = float(np.mean([accuracy(p) for p, _ in runs]))
    hit = steps_to(curve)
    reach = f'under 0.1 at step {hit}' if hit else 'never under 0.1'
    print(f'{name:<30} loss after {steps} steps {curve[-1]:6.3f}   {reach:<22} test {acc:6.1%}')
    return {'hit': hit, 'acc': acc, 'loss': float(curve[-1]), 'curve': curve.tolist()}
