#!/usr/bin/env python3
"""rosbag → LeRobotDataset の変換（経過時間を入力に足す版）。段 2 の案 A。

第 5 回の pai_training/scripts/rosbag2lerobot.py（Apache-2.0 表示のある pai_training に含まれる）の
RosbagToLeRobot を継承し、convert() だけを差し替える。元の変換との違いは 2 つ:

1. 動き出す前の待ちを切り落とす: 指令（腕 4 関節＋グリッパー）が最初に変わったコマを t=0 とし、それより前を捨てる。
2. observation.environment_state を「関節角の写し」から「動き出してからの経過秒 t」（1 値）に置き換える。
   短い記録を最後の姿勢で埋めるコマ（元の変換と同じ）でも t は増え続ける。

使い方（~/pai_training で）:
    pixi run python /home/ubuntu/adv1/adv1_rosbag2lerobot_time.py \
        --bag-dir ~/dataset/rosbags/unazuki --output-dir ~/dataset/lerobot_dataset/crane_plus_unazuki_t \
        --config configs/crane_features_unazuki_t.py
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
from rosbag2lerobot import RosbagToLeRobot  # noqa: E402

CMD_CHANGE = 1e-6  # 腕の指令がこれ以上変わったら「動き出し」


class RosbagToLeRobotTime(RosbagToLeRobot):
    def _episode_arrays(self, episode_dir):
        messages = self._get_messages(episode_dir)
        joint_states = messages['/joint_states']
        arm_cmds = messages['/crane_plus_arm_controller/joint_trajectory']
        gripper_cmds = messages['/crane_plus_gripper_controller/joint_trajectory']
        j_times = np.array([ts for _, ts in joint_states])
        j_pos = np.array([msg.position for msg, _ in joint_states])
        a_times = np.array([ts for _, ts in arm_cmds])
        a_pos = np.array([msg.points[0].positions if msg.points else [0, 0, 0, 0] for msg, _ in arm_cmds])
        g_times = np.array([ts for _, ts in gripper_cmds])
        g_pos = np.array([msg.points[0].positions[0] if (msg.points and len(msg.points[0].positions) > 0) else 0.0
                          for msg, _ in gripper_cmds])
        start_time = max(j_times[0], a_times[0], g_times[0])
        end_time = min(j_times[-1], a_times[-1], g_times[-1])
        fps = self.target_freq or 20
        num_frames = int((end_time - start_time) * fps / 1e9)
        target_times = np.linspace(start_time, end_time, num_frames)
        j = self._sample_and_hold(j_times, j_pos, target_times)
        a = self._sample_and_hold(a_times, a_pos, target_times)
        g = self._sample_and_hold(g_times, g_pos, target_times)
        act = np.hstack([a, g.reshape(-1, 1)])
        # 腕 4 関節＋グリッパーのどれかの指令が最初に変わったコマ（ぱくぱく は腕を動かさないのでグリッパーも見る）
        changed = np.where(np.max(np.abs(act - act[0]), axis=1) > CMD_CHANGE)[0]
        k0 = int(changed[0]) if len(changed) else 0
        return j[k0:], act[k0:], k0

    def convert(self):
        fps = self.target_freq or 20
        episodes = []
        print("\nFirst pass: trimming the wait before the motion...")
        for ep_dir in self.episode_dirs:
            j, act, k0 = self._episode_arrays(ep_dir)
            episodes.append((ep_dir, j, act))
            print(f"Episode {ep_dir}: cut {k0} frames ({k0 / fps:.2f}s), keep {len(j)} frames")
        max_frames = max(len(j) for _, j, _ in episodes)
        print(f"\nLongest episode after trimming: {max_frames} frames")
        for idx, (ep_dir, j, act) in enumerate(episodes):
            for i in range(max_frames):
                k = min(i, len(j) - 1)
                self.dataset.add_frame({
                    "observation.state": torch.tensor(j[k], dtype=torch.float32),
                    "observation.environment_state": torch.tensor([i / fps], dtype=torch.float32),
                    "action": torch.tensor(act[k], dtype=torch.float32),
                    "task": self.task_name,
                })
            self.dataset.save_episode()
            print(f"Episode {idx} saved ({ep_dir.name})")
        return str(self.dataset.root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--fps', type=int, default=20)
    args = ap.parse_args()
    out = RosbagToLeRobotTime(args.bag_dir, args.output_dir, args.config, args.fps).convert()
    print(f"Dataset saved to: {out}")


if __name__ == '__main__':
    main()
