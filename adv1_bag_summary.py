"""rosbag の記録 1 本ずつの要約: 長さ・いちばん動いた関節・動き始めと終わり・最大角・途中の止まり（コンテナの中で ROS を読み込んで使う）。

使い方: python3 adv1_bag_summary.py ~/dataset/rosbags/unazuki
第 5 回の演習の記録を調べたときの要約に、最大角と途中の止まりを足したもの。
"""
import os
import sys

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState

STILL = 1e-4  # これ以下の変化は「止まっている」（rad）

root = sys.argv[1]
eps = sorted(os.listdir(root), key=lambda s: int("".join(c for c in s if c.isdigit()) or -1))
print("count =", len(eps))
for ep in eps:
    path = os.path.join(root, ep)
    if not os.path.exists(os.path.join(path, "metadata.yaml")):
        print(f"{ep:5s} NO_METADATA {os.listdir(path)}")
        continue
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"), rosbag2_py.ConverterOptions("", ""))
    ts, pos, names = [], [], None
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic == "/joint_states":
            m = deserialize_message(data, JointState)
            names = list(m.name)
            ts.append(t / 1e9)
            pos.append(list(m.position))
    if not ts:
        print(f"{ep:5s} NO_JOINT_STATES")
        continue
    t0 = ts[0]
    dur = ts[-1] - t0
    travel = [max(p[i] for p in pos) - min(p[i] for p in pos) for i in range(len(names))]
    j = max(range(len(names)), key=lambda i: travel[i])
    q = [p[j] for p in pos]
    moving = [k for k in range(1, len(ts)) if abs(q[k] - q[k - 1]) > STILL]
    first, last = (ts[moving[0]] - t0, ts[moving[-1]] - t0) if moving else (float("nan"), float("nan"))
    peak_k = max(range(len(q)), key=lambda k: abs(q[k]))
    # 動き始めから終わりまでの間で、止まっていた最長の時間とその角度
    pause, pause_at, k0 = 0.0, float("nan"), None
    if moving:
        for k in range(moving[0], moving[-1] + 1):
            if abs(q[k] - q[k - 1]) <= STILL:
                k0 = k if k0 is None else k0
            elif k0 is not None:
                if ts[k] - ts[k0] > pause:
                    pause, pause_at = ts[k] - ts[k0], q[k0]
                k0 = None
    hand = names.index("crane_plus_joint_hand") if "crane_plus_joint_hand" in names else None
    others = ", ".join(f"{names[i].replace('crane_plus_', '')}={travel[i]:.2f}"
                       for i in range(len(names)) if i not in (j, hand) and travel[i] > 0.02)
    print(
        f"{ep:5s} dur={dur:5.1f}s main={names[j].replace('crane_plus_', '')} "
        f"start={q[0]:+.3f} peak={q[peak_k]:+.3f}@{ts[peak_k] - t0:4.1f}s end={q[-1]:+.3f} "
        f"move {first:4.1f}..{last:4.1f}s idle_after={dur - last:4.1f}s "
        f"max_pause={pause:4.2f}s@{pause_at:+.2f} hand={pos[0][hand]:+.3f}->{pos[-1][hand]:+.3f}"
        + (f" | other: {others}" if others else "")
    )
