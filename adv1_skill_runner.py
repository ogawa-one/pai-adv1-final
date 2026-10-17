#!/usr/bin/env python3
"""adv1_skill_runner: CRANE+ V2 (Gazebo) runs ACT-learned motions in the order given by a text instruction.

Physical AI 応用編 1 最終課題・段 1。お辞儀のモデル 1 つで
「指示 → 動きを選ぶ → 実行 → 終わりを判定 → 直立へ戻す」を通す。

制御ループ（20 Hz・観測 5 値 → ACT → 関節指令）は pai_training/scripts/infer.py
（Apache-2.0 表示のある pai_training に含まれる）を元にしている。

使い方（コンテナの中・~/pai_training で）:
    source /opt/ros/humble/setup.bash
    pixi run python /home/ubuntu/adv1/adv1_skill_runner.py --text "お辞儀して" --log-dir /home/ubuntu/adv1/logs

終了コード: 0 = 全部の動きが終わり直立に戻った／1 = どこかで時間切れ・直立に戻れない／2 = 指示から動きを 1 つも拾えなかった
"""
import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from lerobot.common.policies.act.modeling_act import ACTPolicy

# 記録・学習・推論と同じ順番（関節 3 と 4 が入れ替わっている）
ARM_JOINT_NAMES = ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint4', 'crane_plus_joint3']
GRIPPER_JOINT_NAME = 'crane_plus_joint_hand'

CONTROL_HZ = 20.0
CHUNK_DT = 0.1          # ACT が予測する指令の並びの間隔（秒。学習の delta_timestamps.action が 0.1 秒おき）
MOVE_THRESH = 0.10      # 出発の姿勢からこれ以上動いたら「動き出した」（rad）
STILL_WINDOW = 20       # 直近 20 コマ（1 秒）の
STILL_RANGE = 0.01      # 各関節の振れ幅がこれ未満なら「止まった」（rad）
RETURN_TOL = 0.05       # 往復の動き: 動いた後、出発の姿勢との差（腕 4 関節＋グリッパー）がこれ未満に戻り、
                        # そこで STILL_WINDOW（1 秒）止まったら終わり（途中で出発の姿勢を通り過ぎる動き＝手を振る・ぱくぱく のため）
UPRIGHT_TOL = 0.05      # 全関節の絶対値がこれ未満なら直立（rad）
UPRIGHT_MOVE_S = 2      # 直立へ戻す指令の所要時間（秒）
GRIPPER_OPEN = -0.5236  # 記録のときのグリッパー（テレオペの O＝開。teleop_keyboard.py の GRIPPER_OPEN）
                        # お辞儀のモデルは出発時のグリッパーの値に敏感（−0.533 から始めると 0.134 で止まった）
GRIPPER_TOL = 0.003     # グリッパーが「開」に着いたとみなす差（rad）。−0.533 と −0.524 の差 0.009 を区別できる大きさ
FIXED_STD = 1e-3        # 記録での標準偏差がこれ未満の関節は「その動きでは動かない関節」として、記録の平均を入れる
UPRIGHT_TIMEOUT_S = 15.0


@dataclass
class Skill:
    model_path: str
    keywords: list = field(default_factory=list)
    roundtrip: bool = False  # True: 行って戻る動き（出発の姿勢に戻ったら終わり）／False: 片道（行った先で止まったら終わり）
    time_input: bool = False  # True: environment_state に「動きを始めてからの経過秒」を渡す（adv1_rosbag2lerobot_time.py で変換したモデル）
    chunk_steps: int = 1  # 1 回の予測で実行する指令の数（ACT の n_action_steps）。並びは 0.1 秒おき（学習の delta_timestamps）。
                          # 1 だと毎回「いまの角度」から予測し直し、角度をまねる近道に引きずられることがある（お辞儀）
    timeout_s: float = 60.0


# 動きの台帳。動きを増やすときはここに足す（段 3）。
SKILLS = {
    # 4 つとも、経過時間を入力に入れ・バッチ 16・1,800 ステップ・関節角の正規化の標準偏差に下限 0.1（_f）で学習したモデル。
    # 前のモデル: お辞儀 crane_plus_pekori_act（関節角だけ・200 ステップ。動かない関節の雑音に反応していた）／
    #             うなずく crane_plus_unazuki_t_act／手を振る・ぱく …_t_s1800_b16_act（下限なし）
    'pekori': Skill(
        model_path='/home/ubuntu/checkpoints/train/crane_plus_pekori_t_s1800_b16_f_act',
        keywords=['お辞儀', 'おじぎ', 'ぺこり', 'bow'],
        time_input=True,
        chunk_steps=8,  # 0.8 秒分をまとめて実行（1 だと 0.042 rad で止まった。10/13）
    ),
    # 段 2 の案 A: 経過時間を入力に足したモデル（入力が関節角だけのモデル crane_plus_unazuki_act は直立で動かなかった）
    'unazuki': Skill(
        model_path='/home/ubuntu/checkpoints/train/crane_plus_unazuki_t_s1800_b16_f_act',
        keywords=['うなず', '頷', 'nod'],
        roundtrip=True,
        time_input=True,
    ),
    # 段 3: 関節 2（肩）を 0 → +0.3 → −0.3 → 0（途中で 0 を通る往復）。バッチ 16・1,800 ステップ
    'furu': Skill(
        model_path='/home/ubuntu/checkpoints/train/crane_plus_furu_t_s1800_b16_f_act',
        keywords=['手を振', '振って', 'バイバイ', 'wave'],
        roundtrip=True,
        time_input=True,
    ),
    # 段 3: グリッパーを 1 回閉じて開く。記録は 開→閉→開→閉→開 だが、モデルは 1 回目の閉→開 だけを再現する
    # （「1 回閉じて開く」動きとして採用した）。「ぱくぱく」は 'ぱく' が 2 回当たるので 2 回続けて実行する。
    'paku': Skill(
        model_path='/home/ubuntu/checkpoints/train/crane_plus_pakupaku_t_s1800_b16_f_act',
        keywords=['ぱく', 'パク', 'つかん'],
        roundtrip=True,
        time_input=True,
    ),
}


class KeywordParser:
    """決めた言葉を拾って、文中に出てきた順に動きの名前を並べる。

    LLM 等に差し替えるときは、同じ parse(text) -> list[str] を持つものを渡す。
    """

    def __init__(self, skills):
        self.skills = skills

    def parse(self, text):
        hits = []
        for name, skill in self.skills.items():
            for kw in skill.keywords:
                start = 0
                while True:
                    i = text.find(kw, start)
                    if i < 0:
                        break
                    hits.append((i, len(kw), name))
                    start = i + len(kw)
        # 同じ位置で重なった言葉（例: 'お辞儀' と 'おじぎ' は重ならないが将来のため）は長いほうを残す
        hits.sort(key=lambda h: (h[0], -h[1]))
        seq, covered_to = [], -1
        for pos, length, name in hits:
            if pos < covered_to:
                continue
            seq.append(name)
            covered_to = pos + length
        return seq


class SkillRunner(Node):
    def __init__(self):
        super().__init__('adv1_skill_runner')
        self.arm_pub = self.create_publisher(JointTrajectory, '/crane_plus_arm_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, '/crane_plus_gripper_controller/joint_trajectory', 10)
        self.create_subscription(JointState, '/joint_states', self._on_joint_state, 1)  # 最新の 1 つだけ持つ（読むたびに最新）
        self.state = None
        self.policies = {}
        self.fixed_inputs = {}  # 動きの名前 → (記録で動かなかった関節の印, 記録の平均)
        self.rows = []
        self.t0 = time.monotonic()

    def _on_joint_state(self, msg):
        pos = dict(zip(msg.name, msg.position))
        arm = [pos.get(j, 0.0) for j in ARM_JOINT_NAMES]
        self.state = np.array(arm + [pos.get(GRIPPER_JOINT_NAME, 0.0)], dtype=np.float32)

    def spin_for(self, seconds):
        # 予定より遅れていても（seconds <= 0）、届いた観測は必ず 1 回読む。
        # 10/12、推論が 50 ms を超えて遅れ続けたとき観測を一度も読まず、関節角が出発時のまま固まって動かなかった。
        rclpy.spin_once(self, timeout_sec=0.0)
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=max(0.0, min(0.01, end - time.monotonic())))

    def wait_for_state(self, timeout_s=10.0):
        end = time.monotonic() + timeout_s
        while self.state is None and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.state is not None

    def load(self, name, skill):
        policy = ACTPolicy.from_pretrained(skill.model_path)
        policy.eval()
        policy.to(torch.device('cpu'))
        policy.config.n_action_steps = skill.chunk_steps  # reset() で待ち行列の長さに反映される
        self.policies[name] = policy
        # 記録で動かなかった関節（標準偏差がほぼ 0）は、正規化で 0.001 rad の差が何百倍にも拡大される
        # （10/12 の通しで、前の動きの残り −0.035 rad や グリッパー −0.525 で動きが止まった）。
        # その関節には、いまの値の代わりに記録の平均を入れる（その動きには情報を持たない関節なので、雑音だけを消す）。
        st = policy.normalize_inputs.buffer_observation_state
        mean = st['mean'].detach().cpu().numpy().astype(np.float32)
        fixed = st['std'].detach().cpu().numpy() < FIXED_STD
        self.fixed_inputs[name] = (fixed, mean)
        print(f'[load] {name}: fixed inputs (std<{FIXED_STD}) = {fixed.astype(int).tolist()}', flush=True)

    def _publish(self, arm, gripper=None, duration_s=0.1):
        sec = int(duration_s)
        nsec = int(round((duration_s - sec) * 1e9))
        jt = JointTrajectory()
        jt.joint_names = ARM_JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in arm]
        pt.time_from_start.sec, pt.time_from_start.nanosec = sec, nsec
        jt.points.append(pt)
        self.arm_pub.publish(jt)
        if gripper is not None:
            jg = JointTrajectory()
            jg.joint_names = [GRIPPER_JOINT_NAME]
            pg = JointTrajectoryPoint()
            pg.positions = [float(gripper)]
            pg.time_from_start.sec, pg.time_from_start.nanosec = sec, nsec
            jg.points.append(pg)
            self.gripper_pub.publish(jg)

    def _log(self, step_name, phase, action=None, note=''):
        a = action if action is not None else [float('nan')] * 5
        self.rows.append([round(time.monotonic() - self.t0, 3), step_name, phase,
                          *[round(float(x), 5) for x in self.state], *[round(float(x), 5) for x in a], note])

    def run_skill(self, idx, name, skill):
        """1 つの動きを実行する。戻り値は (結果, 所要秒, 出発の姿勢から最も離れた量)。

        結果: 'settled'（片道で止まった＝成功）／'returned'（往復で出発の姿勢に戻った＝成功）／
              'stalled'（往復なのに途中で止まった＝失敗）／'timeout'（失敗）
        """
        policy = self.policies[name]
        policy.reset()  # 予測の待ち行列を空にする（前の動きの続きを出さない）
        label = f'{idx}:{name}'
        q0 = self.state.copy()
        window = []    # 直近 STILL_WINDOW コマの観測
        a_window = []  # 同じく指令。折り返し点では腕が一瞬止まるが指令は動いているので、両方が止まったときだけ「止まった」とする（手を振る で誤判定した）
        moved = False
        peak = 0.0
        # まとめて実行するときは、並びの 1 個（0.1 秒分）を 20 Hz の 2 回にわたって使う
        ticks_per_action = int(round(CHUNK_DT * CONTROL_HZ)) if skill.chunk_steps > 1 else 1
        tick = 0
        action = None
        period = 1.0 / CONTROL_HZ
        start = time.monotonic()
        next_tick = start
        while True:
            now = time.monotonic()
            if now - start > skill.timeout_s:
                self._log(label, 'skill', note='timeout')
                return 'timeout', now - start, peak
            q = self.state.copy()
            dist = float(np.max(np.abs(q - q0)))  # グリッパーも含める（ぱくぱく は腕を動かさない）
            peak = max(peak, dist)
            fixed, mean = self.fixed_inputs[name]
            q_in = np.where(fixed, mean, q).astype(np.float32)
            env = np.array([now - start], dtype=np.float32) if skill.time_input else q_in
            obs = {
                'observation.state': torch.from_numpy(q_in).unsqueeze(0),
                'observation.environment_state': torch.from_numpy(env).unsqueeze(0),
            }
            if tick % ticks_per_action == 0:
                with torch.no_grad():
                    action = policy.select_action(obs).cpu().numpy().flatten()
            tick += 1
            self._publish(action[:4], action[4], duration_s=0.1)

            if not moved and dist > MOVE_THRESH:
                moved = True
                self._log(label, 'skill', action, note='moved')
            else:
                self._log(label, 'skill', action)
            if moved:
                window.append(q)
                a_window.append(action)
                if len(window) > STILL_WINDOW:
                    window.pop(0)
                    a_window.pop(0)
                if (len(window) == STILL_WINDOW
                        and np.max(np.ptp(np.stack(window), axis=0)) < STILL_RANGE
                        and np.max(np.ptp(np.stack(a_window), axis=0)) < STILL_RANGE):
                    # 往復の動きは「出発の姿勢で止まった」ときだけ終える。ほかの場所で止まって見えても、
                    # ゆっくり戻る途中のことがある（手を振る）ので続け、上限時間で打ち切る（結果 'timeout'）。
                    if not skill.roundtrip or dist < RETURN_TOL:
                        result = 'returned' if skill.roundtrip else 'settled'
                        self._log(label, 'skill', action, note=result)
                        return result, time.monotonic() - start, peak

            next_tick += period
            self.spin_for(max(0.0, next_tick - time.monotonic()))

    def at_start_pose(self):
        """腕が直立で、グリッパーが記録のときの「開」に近いか。"""
        return (np.max(np.abs(self.state[:4])) < UPRIGHT_TOL
                and abs(self.state[4] - GRIPPER_OPEN) < GRIPPER_TOL)

    def return_upright(self, label):
        """腕を直立（全関節 0）へ、グリッパーを記録のときの「開」へ戻す。戻り値は (着いたか, 所要秒)。"""
        self._publish([0.0, 0.0, 0.0, 0.0], gripper=GRIPPER_OPEN, duration_s=UPRIGHT_MOVE_S)
        start = time.monotonic()
        while time.monotonic() - start < UPRIGHT_TIMEOUT_S:
            self.spin_for(1.0 / CONTROL_HZ)
            self._log(label, 'upright')
            if self.at_start_pose():
                self._log(label, 'upright', note='reached')
                return True, time.monotonic() - start
        self._log(label, 'upright', note='timeout')
        return False, time.monotonic() - start

    def write_log(self, path):
        head = ['t', 'step', 'phase',
                *[f'q_{j}' for j in ARM_JOINT_NAMES], 'q_hand',
                *[f'a_{j}' for j in ARM_JOINT_NAMES], 'a_hand', 'note']
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(head)
            w.writerows(self.rows)


def main():
    ap = argparse.ArgumentParser(description='Run ACT-learned CRANE+ V2 motions in the order given by a text instruction')
    ap.add_argument('--text', required=True, help='指示の文（例: "お辞儀して"）')
    ap.add_argument('--log-dir', default='.', help='1 歩ごとの記録（CSV）を書く場所')
    ap.add_argument('--events', default='', help='動きの始まり・終わりを壁時計の時刻で追記するファイル（動画の字幕用。adv1_make_video.py が読む）')
    args = ap.parse_args()

    def event(kind, *fields):
        if args.events:
            with open(args.events, 'a') as f:
                f.write('\t'.join([f'{time.time():.3f}', kind, *[str(x) for x in fields]]) + '\n')

    parser = KeywordParser(SKILLS)
    seq = parser.parse(args.text)
    print(f'[parse] text={args.text!r} -> {seq}', flush=True)
    event('parse', args.text, ','.join(seq))
    if not seq:
        print('[parse] 知っている動きの言葉が見つかりません。動かさずに終わります。'
              f' 知っている言葉: { {n: s.keywords for n, s in SKILLS.items()} }', flush=True)
        return 2

    rclpy.init()
    node = SkillRunner()
    rc = 0
    try:
        for name in sorted(set(seq)):
            node.load(name, SKILLS[name])
        if not node.wait_for_state():
            print('[error] /joint_states が届きません（Gazebo は起動していますか）', flush=True)
            return 1
        # 腕の指令を出すのが自分だけか確かめる（10/12、記録に使ったテレオペが動いたままで、20 Hz の直立指令に上書きされた）
        node.spin_for(1.0)  # 他のノードの発見を待つ
        others = [i.node_name for i in node.get_publishers_info_by_topic('/crane_plus_arm_controller/joint_trajectory')
                  if i.node_name != node.get_name()]
        if others:
            print(f'[error] ほかのノードが腕に指令を出しています: {others}（テレオペを Ctrl+C で止めてから動かす）', flush=True)
            return 1
        print(f'[start] state={np.round(node.state, 3).tolist()}', flush=True)
        if not node.at_start_pose():
            ok, dt = node.return_upright('0:init')
            print(f'[init] upright reached={ok} {dt:.1f}s', flush=True)
            if not ok:
                return 1
        for i, name in enumerate(seq, 1):
            event('skill_start', i, len(seq), name)
            result, dt, peak = node.run_skill(i, name, SKILLS[name])
            event('skill_end', i, len(seq), name, result)
            q_end = np.round(node.state, 3).tolist()
            print(f'[skill {i}/{len(seq)}] {name}: {result} {dt:.1f}s peak={peak:.3f} end={q_end}', flush=True)
            ok, dt_up = node.return_upright(f'{i}:{name}')
            print(f'[skill {i}/{len(seq)}] upright reached={ok} {dt_up:.1f}s', flush=True)
            event('upright_end', i, len(seq), name, ok)
            if result not in ('settled', 'returned') or not ok:
                rc = 1
                break
    finally:
        os.makedirs(args.log_dir, exist_ok=True)
        path = os.path.join(args.log_dir, time.strftime('adv1_run_%y%m%d_%H%M%S.csv'))
        node.write_log(path)
        print(f'[log] {path} rows={len(node.rows)}', flush=True)
        node.destroy_node()
        rclpy.shutdown()
    print(f'[done] rc={rc}', flush=True)
    return rc


if __name__ == '__main__':
    sys.exit(main())
