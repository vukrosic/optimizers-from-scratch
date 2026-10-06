"""Step 8. The race: every optimizer, the same network, the same pictures, every learning rate.

Each optimizer is tried with 17 learning rates from 0.0001 to 1, five starting weights each,
400 steps per run. That is 425 trainings; they run on all processor cores and take a minute or two.
"""
from multiprocessing import Pool

import numpy as np

from net import accuracy, steps_to, train, typical
from step2_sgd import SGD
from step3_momentum import Momentum
from step4_rmsprop import RMSProp
from step5_adam import Adam
from step7_muon import Muon

RACERS = {'SGD': SGD, 'momentum': Momentum, 'RMSProp': RMSProp, 'Adam': Adam, 'Muon': Muon}
RATES = [float(f'{r:.3g}') for r in np.geomspace(0.0001, 1, 17)]
SEEDS = 5
GOOD = 0.1                                           # a learning rate "works" if the typical run ends under this loss


def one(job):
    name, lr, seed = job
    params, curve = train(RACERS[name](lr), 400, seed)
    return curve, accuracy(params)


def race():
    jobs = [(name, lr, seed) for name in RACERS for lr in RATES for seed in range(SEEDS)]
    with Pool() as pool:
        done = pool.map(one, jobs)
    table = {}
    for name in RACERS:
        rows = []
        for lr in RATES:
            runs = [d for j, d in zip(jobs, done) if j[:2] == (name, lr)]
            curve = typical([c for c, _ in runs])
            rows.append({'lr': lr, 'loss': float(curve[-1]), 'hit': steps_to(curve),
                         'acc': float(np.mean([a for _, a in runs])), 'curve': curve.tolist()})
        table[name] = rows
    return table


def run():
    table = race()
    print(f'{"":<10}{"best rate":>10}{"final loss":>12}{"under 0.1 at":>14}{"test":>8}{"rates that work":>18}')
    out = {}
    for name, rows in table.items():
        best = min(rows, key=lambda r: r['loss'])
        works = [r['lr'] for r in rows if r['loss'] < GOOD]
        hit = f'step {best["hit"]}' if best['hit'] else 'never'
        print(f'{name:<10}{best["lr"]:>10}{best["loss"]:>12.3f}{hit:>14}{best["acc"]:>8.1%}{len(works):>12} of {len(rows)}')
        out[name] = {'best': best, 'works': works, 'rows': rows}
    print('final loss at every learning rate (. = over 1):')
    print(' ' * 10 + ''.join(f'{lr:>9}' for lr in RATES))
    for name, rows in table.items():
        print(f'{name:<10}' + ''.join(f'{r["loss"]:9.2f}' if r['loss'] < 1 else f'{".":>9}' for r in rows))
    slow, quick = train(SGD(0.01), 400)[0], train(Adam(0.001), 400)[0]
    print(f'library default rates, no search:  SGD 0.01 -> test {accuracy(slow):.1%},  Adam 0.001 -> test {accuracy(quick):.1%}')
    out['default'] = (accuracy(slow), accuracy(quick))
    return out


if __name__ == '__main__':
    run()
