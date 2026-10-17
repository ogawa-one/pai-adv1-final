#!/usr/bin/env python3
"""経過時間を入れたモデルが、時刻ごとに何を予測しているかを、記録の指令（中央値）と並べて見る。

1. 時刻 t ごとに、全エピソードの記録の指令（--joint 番目）の中央値・最小・最大。
2. 同じ t で、関節角を「各エピソードのその時刻の関節角」に入れ替えたときの予測（最小・最大）。
   予測の幅が 0 なら、モデルは関節角を見ずに時刻だけで決めている。

使い方（~/pai_training で）:
    pixi run python /home/ubuntu/adv1/adv1_diag_time.py --model <モデル> --dataset <データセット> --joint 1
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch

from lerobot.common.policies.act.modeling_act import ACTPolicy

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--dataset', required=True)
ap.add_argument('--joint', type=int, required=True)
ap.add_argument('--step', type=float, default=0.5, help='何秒おきに見るか')
args = ap.parse_args()

policy = ACTPolicy.from_pretrained(os.path.expanduser(args.model))
policy.eval()
eps = []
for f in sorted(glob.glob(os.path.join(os.path.expanduser(args.dataset), 'data/chunk-000/*.parquet'))):
    d = pd.read_parquet(f)
    eps.append((np.stack(d['observation.state'].values).astype(np.float32),
                np.stack(d['observation.environment_state'].values).astype(np.float32).reshape(len(d), -1),
                np.stack(d['action'].values)))
fps = 20
n = min(len(s) for s, _, _ in eps)
j = args.joint
print(f'model={args.model} episodes={len(eps)} frames={n} joint={j}')
print('    t | record median [min, max] | pred over episodes\' states [min, max]')
for i in range(0, n, max(1, int(args.step * fps))):
    rec = np.array([a[i, j] for _, _, a in eps])
    preds = []
    for s, env, _ in eps:
        policy.reset()
        with torch.no_grad():
            out = policy.select_action({'observation.state': torch.from_numpy(s[i:i + 1]),
                                        'observation.environment_state': torch.from_numpy(env[i:i + 1])})
        preds.append(out.numpy().ravel()[j])
    preds = np.array(preds)
    print(f'{i / fps:5.2f} | {np.median(rec):+.3f} [{rec.min():+.3f}, {rec.max():+.3f}] | '
          f'{np.median(preds):+.3f} [{preds.min():+.3f}, {preds.max():+.3f}]')
