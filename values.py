"""Runs every step once more and saves the numbers and curves the film draws: values.json."""
import contextlib
import io
import json

import numpy as np

import step1_gradient_descent as s1
import step2_sgd as s2
import step3_momentum as s3
import step4_rmsprop as s4
import step5_adam as s5
import step6_adamw as s6
import step7_muon as s7
import step8_race as s8
from net import SIZES, XTE, XTR, YTE, YTR, loss_and_grads, new_weights

R = lambda a, k=3: np.round(np.asarray(a, dtype=float), k).tolist()


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def main():
    V = {'data': {'n_train': len(XTR), 'n_test': len(XTE), 'samples': R(XTR[:40], 2), 'labels': YTR[:40].tolist(),
                  'sizes': SIZES, 'weights': int(sum(p.size for p in new_weights()))}}
    g = quiet(s1.run)
    V['gd'] = {'start': list(s1.START), 'paths': {str(k): R(v[:400]) for k, v in g.items()},
               'steps': {str(k): len(v) - 1 for k, v in g.items()}, 'loss_end': {str(k): float(s1.loss(v[-1])) for k, v in g.items()},
               'nudge': [round(s1.nudge(s1.START, 0), 3), round(s1.nudge(s1.START, 1), 3)]}
    r = quiet(s2.run)
    V['sgd'] = {'full': R(r['full']), 'mini': R(r['mini']), 'full_acc': r['full_acc'], 'mini_acc': r['mini_acc'],
                'mini_steps': r['mini_steps'], 'res': r['res']}
    r = quiet(s3.run)
    V['mom'] = {'plain': R(r['plain']), 'fast': R(r['fast']), 'works_plain': int(r['works_plain']),
                'works_fast': int(r['works_fast']), 'sgd': r['sgd'], 'mom': r['mom']}
    r = quiet(s4.run)
    _, grads = loss_and_grads(new_weights(), XTR, YTR)
    mags = np.abs(np.concatenate([x.ravel() for x in grads]))
    hist = np.histogram(np.log10(mags[mags > 0]), bins=24, range=(-8, -1))[0]
    V['rms'] = {'path': R(r['path'][:400]), 'loss100': float(s1.loss(r['path'][100])), 'loss1000': float(s1.loss(r['path'][-1])),
                'sizes': R(r['sizes'], 4), 'ratio': r['ratio'], 'hist': hist.tolist(), 'zero': int((mags == 0).sum()),
                'mom': r['mom'], 'rms': r['rms']}
    r = quiet(s5.run)
    V['adam'] = {'path': R(r['path']), 'works': r['works'], 'raw': R(r['raw']), 'fixed': R(r['fixed']), 'rms': r['rms'], 'adam': r['adam'],
                 'steps': {str(lr): len(s1.descend(s5.Adam(lr))) - 1 for lr in (0.1, 0.3, 1.0)}}
    r = quiet(s6.run)
    trace = {}
    for key, make in (('old', lambda: s6.AdamL2(0.01, 0.1)), ('new', lambda: s6.AdamW(0.01, 0.1))):
        w, opt, rows = np.ones((1, 3)), make(), []
        for t in range(200):
            opt.step([w], [np.array([[1.0, 0.1, 0.01]]) * (1 if t % 2 == 0 else -1)])
            if t % 4 == 3:
                rows.append(R(w[0]))
        trace[key] = rows
    V['adamw'] = {'old': R(r['old']), 'new': R(r['new']), 'trace': trace, 'wrong': s6.noisy()[2],
                  'fit': {k: [float(v[0]), float(v[1])] for k, v in r.items() if k not in ('old', 'new')}}
    r = quiet(s7.run)
    _, grads = loss_and_grads(new_weights(), XTR[:32], YTR[:32])
    G = grads[4]
    V['muon'] = {'before': R(r['before'], 5), 'after': R(r['after']), 'rounds': [R(x) for x in r['rounds']],
                 'G': R(G / np.abs(G).max(), 2), 'O': R(s7.orthogonalize(G) / np.abs(s7.orthogonalize(G)).max(), 2),
                 'above': [int((r['before'] > 0.1).sum()), int((r['after'] > 0.1 * r['after'].max()).sum())],
                 'adam': r['adam'], 'muon': r['muon']}
    r = quiet(s8.run)
    V['race'] = {'rates': s8.RATES, 'default': list(r['default']),
                 'opt': {k: {'best': {kk: vv for kk, vv in v['best'].items()}, 'works': v['works'],
                             'loss': [round(min(x['loss'], 9), 3) for x in v['rows']], 'acc': R([x['acc'] for x in v['rows']])}
                         for k, v in r.items() if k != 'default'}}
    json.dump(V, open('values.json', 'w'))
    print('values.json', len(json.dumps(V)), 'chars')


if __name__ == '__main__':
    main()
