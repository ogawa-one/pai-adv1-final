#!/bin/bash
# 1 つの動き（例 unazuki）を、rosbag → LeRobot 形式への変換 → ACT の学習まで通す。コンテナ pai_ros2 の中で使う。
# 使い方: bash /home/ubuntu/adv1/adv1_train_skill.sh unazuki          # 第 5 回と同じ変換（入力は関節角だけ）
#         bash /home/ubuntu/adv1/adv1_train_skill.sh unazuki time     # 経過時間を入力に足す変換（adv1_rosbag2lerobot_time.py）
#
# 第 5 回の pai_training は設定ファイルの場所が固定（configs/conversion_config.yaml・configs/train_config.yaml）なので、
# 動きごとの設定をそこへ一時的に置き、終わったら（失敗しても）元の設定（お辞儀用）に戻す。
# 学習の設定は 第 5 回のお辞儀と同じ。変えるのは データの場所・名前・出力先（と time のときの特徴量）だけ。
export PATH=/home/ubuntu/.pixi/bin:$PATH
source /opt/ros/humble/setup.bash  # set -u より前に読む（ROS の setup.bash は未定義の変数を参照する）
set -u
SKILL=$1
MODE=${2:-plain}
STEPS=${3:-}   # 学習のステップ数（省略時は train_config.yaml のまま＝200）。指定すると名前に _s<STEPS> が付く
BATCH=${4:-}   # バッチの大きさ（省略時は train_config.yaml のまま＝2）。指定すると名前に _b<BATCH> が付く
FLOOR=${5:-}   # 関節角の正規化に使う標準偏差の下限（rad。例 0.1）。指定すると名前に _f が付く
               # 記録で動かない関節は標準偏差がほぼ 0 で、0.001 rad の差が何百倍以上に拡大される。変換の後、学習の前に
               # meta/episodes_stats.jsonl（LeRobot v2.1 は学習時にここから正規化の値を集計する）の std を下限で持ち上げる
case "$MODE" in
  plain) NAME=$SKILL ;;
  time)  NAME=${SKILL}_t ;;
  *) echo "[error] unknown mode: $MODE"; exit 94 ;;
esac
[ -n "$STEPS" ] && NAME=${NAME}_s$STEPS
[ -n "$BATCH" ] && NAME=${NAME}_b$BATCH
[ -n "$FLOOR" ] && NAME=${NAME}_f
PT=/home/ubuntu/pai_training
BAGS=/home/ubuntu/dataset/rosbags/$SKILL
DS=/home/ubuntu/dataset/lerobot_dataset/crane_plus_$NAME
OUT=/home/ubuntu/checkpoints/train/crane_plus_${NAME}_act
LOG=/home/ubuntu/adv1/train_$NAME
FEAT=configs/crane_features_$NAME.py
mkdir -p "$LOG"

[ -d "$BAGS" ] || { echo "[error] no bags: $BAGS"; exit 91; }
[ -e "$DS" ] && { echo "[error] dataset already exists: $DS (消すか名前を変える)"; exit 92; }
[ -e "$OUT" ] && { echo "[error] checkpoint already exists: $OUT (消すか名前を変える)"; exit 93; }

cd "$PT" || exit 90
cp configs/conversion_config.yaml configs/conversion_config.yaml.adv1bak
cp configs/train_config.yaml configs/train_config.yaml.adv1bak
restore() {
  mv -f configs/conversion_config.yaml.adv1bak configs/conversion_config.yaml
  mv -f configs/train_config.yaml.adv1bak configs/train_config.yaml
  echo "[restore] configs back to: $(grep bag_dir configs/conversion_config.yaml) / $(grep '^repo_id' configs/train_config.yaml)"
}
trap restore EXIT

sed -e "s|^repo_id = .*|repo_id = \"crane_plus_$NAME\"|" \
    -e "s|^root = .*|root = \"$DS\"|" \
    -e "s|^task_name = .*|task_name = \"$SKILL\"|" configs/crane_features.py > "$FEAT"
if [ "$MODE" = time ]; then
  # environment_state を 1 値（動き出してからの経過秒）にする
  python3 - "$FEAT" <<'EOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
s, n = re.subn(r'("observation\.environment_state": \{\s*"dtype": "float32",\s*)"shape": \(5,\),\s*"names": \[[^\]]*\]',
               r'\1"shape": (1,),\n        "names": ["t"]', s)
assert n == 1, f"environment_state block not found ({n})"
open(p, 'w').write(s)
EOF
  [ $? -eq 0 ] || exit 95
fi
cat > configs/conversion_config.yaml <<EOF
bag_dir: "$BAGS"
output_dir: "$DS"
config: "$FEAT"
target_freq: 20
EOF
sed -e "s|^dataset_root: .*|dataset_root: \"$DS\"|" \
    -e "s|^repo_id: .*|repo_id: \"crane_plus_$NAME\"|" \
    -e "s|^output_directory: .*|output_directory: \"$OUT\"|" \
    -e "s|^config_path: .*|config_path: \"$FEAT\"|" configs/train_config.yaml.adv1bak > configs/train_config.yaml
if [ -n "$STEPS" ]; then
  sed -i "s|^training_steps: .*|training_steps: $STEPS|" configs/train_config.yaml
  grep -q "^training_steps: $STEPS$" configs/train_config.yaml || { echo "[error] training_steps not set"; exit 97; }
fi
if [ -n "$BATCH" ]; then
  sed -i "s|^batch_size: .*|batch_size: $BATCH|" configs/train_config.yaml
  grep -q "^batch_size: $BATCH$" configs/train_config.yaml || { echo "[error] batch_size not set"; exit 98; }
fi
echo "[config] $(grep -E '^(training_steps|batch_size|output_directory):' configs/train_config.yaml | tr '\n' ' ')"
cp configs/conversion_config.yaml configs/train_config.yaml "$FEAT" "$LOG/"
grep -A3 '"observation.environment_state"' "$FEAT" | tr -s ' ' | tr '\n' ' '; echo

echo "[convert:$MODE] $BAGS -> $DS"
T0=$(date +%s)
if [ "$MODE" = time ]; then
  pixi run python /home/ubuntu/adv1/adv1_rosbag2lerobot_time.py --bag-dir "$BAGS" --output-dir "$DS" --config "$FEAT" > "$LOG/convert.log" 2>&1
else
  pixi run convert > "$LOG/convert.log" 2>&1
fi
RC=$?
echo "[convert] rc=$RC $(( $(date +%s) - T0 ))s"
grep -E "^Episode .*(cut|frames)|Longest" "$LOG/convert.log"
[ $RC -eq 0 ] || { tail -20 "$LOG/convert.log"; exit $RC; }

if [ -n "$FLOOR" ]; then
  # 関節角（observation.state と、関節角の写しのときの environment_state）の std を下限で持ち上げる。経過時間（1 値）は触らない
  python3 - "$DS/meta/episodes_stats.jsonl" "$FLOOR" <<'EOF'
import json, sys
path, floor = sys.argv[1], float(sys.argv[2])
eps = [json.loads(line) for line in open(path)]
changed, low = 0, []
for ep in eps:
    for key in ("observation.state", "observation.environment_state"):
        st = ep["stats"].get(key)
        if st is None or len(st["std"]) < 2:
            continue
        new = [max(x, floor) for x in st["std"]]
        changed += sum(a != b for a, b in zip(new, st["std"]))
        st["std"] = new
        low.append(min(new))
open(path, "w").write("".join(json.dumps(ep) + "\n" for ep in eps))
print(f"[floor] std>={floor} episodes={len(eps)} raised_values={changed} min_std_after={min(low):.3f}")
EOF
  [ $? -eq 0 ] || exit 99
fi

echo "[train] -> $OUT"
rm -f "$LOG/mem_mb.log" "$LOG/.done"
( while [ ! -f "$LOG/.done" ]; do free -m | sed -n 2p | awk '{print $3}' >> "$LOG/mem_mb.log"; sleep 5; done ) &
T0=$(date +%s)
if [ "$MODE" = time ]; then
  # 1 値の environment_state はデータセットで (B, 1) になり、train.py 125 行目の [:, -1, :]（3 次元前提）で落ちる。
  # 第 5 回の train.py は書き換えず、その 1 行だけを「3 次元のときだけ取り出す」に差し替えた写しを毎回作って使う。
  TRAIN_PY="$LOG/adv1_train_time.py"
  sed 's|batch\["observation.environment_state"\] = batch\["observation.environment_state"\]\[:, -1, :\]|_e = batch["observation.environment_state"]; batch["observation.environment_state"] = _e[:, -1, :] if _e.dim() == 3 else _e|' \
    scripts/train.py > "$TRAIN_PY"
  N=$(grep -c '_e.dim() == 3' "$TRAIN_PY")
  [ "$N" -eq 1 ] || { echo "[error] train.py patch applied $N times (expected 1)"; exit 96; }
  diff scripts/train.py "$TRAIN_PY" | grep -c '^[<>]' | sed 's/^/[train] patched lines (old+new): /'
  WANDB_MODE=offline pixi run python "$TRAIN_PY" > "$LOG/train.log" 2>&1
else
  WANDB_MODE=offline pixi run train > "$LOG/train.log" 2>&1
fi
RC=$?
touch "$LOG/.done"
echo "[train] rc=$RC $(( $(date +%s) - T0 ))s mem_max_mb=$(sort -n "$LOG/mem_mb.log" | tail -1)"
grep -E "input_features" "$LOG/train.log" | cut -c1-300
grep -E "^step: " "$LOG/train.log" | sed -n '1p;$p'
[ $RC -eq 0 ] || tail -20 "$LOG/train.log"
ls -l "$OUT" 2>/dev/null
exit $RC
