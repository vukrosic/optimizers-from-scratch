"""Quick checks for every step. Run: python3 tests.py  (about 5 seconds)"""
import numpy as np

from net import XTR, YTR, loss_and_grads, new_weights, steps_to, train
from step1_gradient_descent import START, descend, grad, gradient_descent, loss, nudge
from step2_sgd import SGD
from step3_momentum import Momentum
from step4_rmsprop import RMSProp
from step5_adam import Adam, first_steps
from step6_adamw import AdamL2, AdamW, shrink
from step7_muon import Muon, orthogonalize

# step 1: the formula agrees with nudging; a safe rate goes down, a rate past 0.02 blows up
assert abs(nudge(START, 0) - grad(START)[0]) < 1e-3 and abs(nudge(START, 1) - grad(START)[1]) < 1e-3
assert loss(gradient_descent(0.019)[-1]) < 0.01 and loss(gradient_descent(0.021)[-1]) > 1e6

# the network: backpropagation agrees with nudging a weight
params = new_weights()
base, grads = loss_and_grads(params, XTR[:20], YTR[:20])
for layer, at in ((0, (30, 5)), (4, (3, 7)), (10, (2, 1)), (5, (4,))):
    params[layer][at] += 1e-5
    slope = (loss_and_grads(params, XTR[:20], YTR[:20])[0] - base) / 1e-5
    params[layer][at] -= 1e-5
    assert abs(slope - grads[layer][at]) < 1e-4, (layer, slope, grads[layer][at])

# step 2: one SGD step lowers the loss on the pictures it looked at
before = loss_and_grads(params, XTR[:32], YTR[:32])[0]
SGD(0.05).step(params, loss_and_grads(params, XTR[:32], YTR[:32])[1])
assert loss_and_grads(params, XTR[:32], YTR[:32])[0] < before

# step 3: momentum with beta 0 is plain SGD; with 0.9 it crosses the valley faster
a, b = descend(SGD(0.01), 20), descend(Momentum(0.01, beta=0.0), 20)
assert np.allclose(a, b)
assert len(descend(Momentum(0.019))) < len(descend(SGD(0.019)))

# step 4: the first RMSProp step has the same size for a steep and a flat weight
w = np.array(START)
RMSProp(0.1).step([w], [grad(w)])
assert abs(abs(w[0] - START[0]) - abs(w[1] - START[1])) < 1e-6

# step 5: with the fix every early Adam step is one learning rate long
assert np.allclose(first_steps(True), 1.0) and first_steps(False)[0] > 3

# step 6: AdamW shrinks every weight by the same share; the old way does not
new, old = shrink(lambda: AdamW(0.01, 0.1)), shrink(lambda: AdamL2(0.01, 0.1))
assert max(new) - min(new) < 0.01 and old[0] - old[2] > 0.5

# step 7: after orthogonalizing, the strengths are all close to 1 and the directions are kept
G = np.random.default_rng(0).normal(size=(32, 32)) * np.linspace(1, 0.01, 32)
s = np.linalg.svd(orthogonalize(G), compute_uv=False)
assert (s > 0.6).sum() >= 31 and s.max() < 1.3      # strengths went from 1 ... 0.001 to about 1
U, _, Vt = np.linalg.svd(G)
assert np.diag(U.T @ orthogonalize(G) @ Vt.T).min() > 0.1
assert orthogonalize(np.ones((64, 32))).shape == (64, 32)

# every optimizer trains the network: 100 steps cut the loss to under a third
start = loss_and_grads(new_weights(), XTR, YTR)[0]
for opt in (SGD(0.1), Momentum(0.0178), RMSProp(0.00316), Adam(0.01), AdamW(0.01, 0.1), Muon(0.00562)):
    curve = train(opt, 100)[1]
    assert curve[-1] < start / 3, (type(opt).__name__, curve[-1])
assert steps_to([0.5, 0.2, 0.09], every=10) == 30 and steps_to([0.5]) is None

print('all tests pass')
