#!/usr/bin/env python3
"""学習したモデルが、学習に使った記録の動きを再現できるかを Gazebo なしで確かめる（段 3 の最初の門）。

記録の各コマの（関節角・経過時間）をモデルに入れ、出てきた指令を記録の指令と比べる。
学習に使ったコマでも再現できなければ、Gazebo で回す前に学習の問題とわかる。

使い方（~/pai_training で）:
    pixi run python /home/ubuntu/adv1/adv1_offline_check.py \
        --model ~/checkpoints/train/crane_plus_furu_t_s3000_act \
        --dataset ~/dataset/lerobot_dataset/crane_plus_furu_t_s3000 --joint 1

--joint は指令の何番目を見るか（0..3 は 関節 1, 2, 4, 3 の順・4 はグリッパー）。
出力: エピソードごとの 平均絶対誤差・記録と予測の最小／最大、と 1 本目の時刻ごとの比較。
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
ap.add_argument('--every', type=int, default=6, help='1 本目の比較を何コマおきに出すか')
args = ap.parse_args()

policy = ACTPolicy.from_pretrained(os.path.expanduser(args.model))
policy.eval()
files = sorted(glob.glob(os.path.join(os.path.expanduser(args.dataset), 'data/chunk-000/*.parquet')))
print(f'model={args.model} episodes={len(files)} joint={args.joint}')
j = args.joint
maes = []
for k, f in enumerate(files):
    d = pd.read_parquet(f)
    s = np.stack(d['observation.state'].values).astype(np.float32)
    env = np.stack(d['observation.environment_state'].values).astype(np.float32).reshape(len(s), -1)
    a = np.stack(d['action'].values)
    pred = np.zeros(len(s))
    for i in range(len(s)):
        policy.reset()  # 1 コマずつ独立に（前のコマの予測を持ち越さない）
        with torch.no_grad():
            out = policy.select_action({'observation.state': torch.from_numpy(s[i:i + 1]),
                                        'observation.environment_state': torch.from_numpy(env[i:i + 1])})
        pred[i] = out.numpy().ravel()[j]
    mae = float(np.mean(np.abs(pred - a[:, j])))
    maes.append(mae)
    print(f'  ep{k}: mae={mae:.3f} true[min,max]=[{a[:, j].min():+.3f},{a[:, j].max():+.3f}] '
          f'pred[min,max]=[{pred.min():+.3f},{pred.max():+.3f}]')
    if k == 0:
        first = [(env[i, 0], a[i, j], pred[i]) for i in range(0, len(s), args.every)]
print(f'mean mae={np.mean(maes):.3f}')
print('ep0 by time (t true pred):')
for t, tr, pr in first:
    print(f'  t={t:5.2f} true={tr:+.3f} pred={pr:+.3f}')
