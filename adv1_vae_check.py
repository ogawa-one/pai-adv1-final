#!/usr/bin/env python3
"""ACT が「学習のときだけ見える答えの要約（VAE の潜在変数）」に頼っていないかを確かめる。

同じモデル・同じコマで、指令の誤差（正規化した L1）を 2 通りで測って比べる:
  train: 学習と同じ（本当の指令の並びを要約器に入れ、その要約を使う）
  eval : 動かすときと同じ（要約は 0）
train の誤差が小さく eval が大きければ、モデルは要約に頼っていて、観測（関節角・時間）から指令を出せていない。

使い方（~/pai_training で）:
    pixi run python /home/ubuntu/adv1/adv1_vae_check.py --model <モデル> --dataset <データセット> --repo-id <名前>
"""
import argparse
import os

import torch
import yaml

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.modeling_act import ACTPolicy

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--dataset', required=True)
ap.add_argument('--repo-id', required=True)
ap.add_argument('--batches', type=int, default=20)
args = ap.parse_args()

# 学習と同じ取り出し方（configs/train_config.yaml の delta_timestamps。第 5 回のまま）
delta = yaml.safe_load(open('configs/train_config.yaml'))['delta_timestamps']
ds = LeRobotDataset(args.repo_id, root=os.path.expanduser(args.dataset), delta_timestamps=delta)
dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(0))
policy = ACTPolicy.from_pretrained(os.path.expanduser(args.model))


def l1(batch, use_latent):
    """正規化した指令の L1 誤差（パディングを除く）。use_latent=True で学習と同じ経路。"""
    policy.train(use_latent)  # ACT は training のときだけ要約器を使い、eval では要約を 0 にする
    b = policy.normalize_inputs(dict(batch))
    b = policy.normalize_targets(b)
    with torch.no_grad():
        actions_hat, _ = policy.model(b)
    err = (b['action'] - actions_hat).abs()
    return (err * ~b['action_is_pad'].unsqueeze(-1)).mean().item()


tr, ev = [], []
for i, batch in enumerate(dl):
    if i >= args.batches:
        break
    for k in ('observation.state', 'observation.environment_state'):
        if batch[k].dim() == 3:
            batch[k] = batch[k][:, -1, :]
    tr.append(l1(batch, True))
    ev.append(l1(batch, False))
print(f'model={args.model}')
print(f'L1 (normalized) train-path={sum(tr) / len(tr):.3f}  eval-path={sum(ev) / len(ev):.3f}  batches={len(tr)}')
