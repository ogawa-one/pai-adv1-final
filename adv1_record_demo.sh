#!/bin/bash
# 段 4: 言葉の指示で 4 つの動きを順に実行する様子を、Gazebo の 3D の部分だけ録画する（字幕は adv1_make_video.py で後から入れる）。
# コンテナ pai_ros2 の中で使う。録画中は localhost:6080 の画面に触れない（窓が動くと録る範囲がずれる）。
# 使い方: bash /home/ubuntu/adv1/adv1_record_demo.sh /home/ubuntu/adv1/video/take1
# 出力: <出力先>/raw.mp4（字幕なし）・events.tsv（動画の開始時刻と、動きの始まり・終わり）・run.log
export PATH=/home/ubuntu/.pixi/bin:$PATH
export DISPLAY=:0
source /opt/ros/humble/setup.bash
set -u
OUT=$1
mkdir -p "$OUT"
EV=$OUT/events.tsv
: > "$EV"
: > "$OUT/run.log"

# 指示の文（10/12 に決めたもの）。順に流す。
TEXTS=("お辞儀して" "手を振って、うなずいて、つかんで" "ぱくぱくしてからお辞儀して" "ジャンプして")
# 録る範囲: Gazebo の窓（1000x845）のうち 3D の部分だけ（窓の左上から +0,+145・580x620）。
# Gazebo を起動し直すと窓の位置が変わる（10/12 は +184,195、10/13 は +1,62）ので、録るたびに xwininfo で窓の位置を読む
CROP_SIZE=580x620
WIN=$(xwininfo -root -tree | grep '"Gazebo": ("ign-gazebo-gui"' | head -1)
WX=$(echo "$WIN" | awk '{print $NF}' | cut -d+ -f2)
WY=$(echo "$WIN" | awk '{print $NF}' | cut -d+ -f3)
[ -n "$WX" ] && [ -n "$WY" ] || { echo "[error] Gazebo window not found"; exit 1; }
CROP_AT=:0.0+$WX,$((WY + 145))
echo "[crop] window +$WX,$WY -> $CROP_SIZE at $CROP_AT"
TITLE_S=4   # 冒頭の題名を出す秒数（この間は直立のまま）
GAP_S=2     # 指示と指示の間

# 腕に指令を出すのが自分だけか（テレオペが動いていると上書きされる）
if [ "$(pgrep -c -f '[t]eleop_keyboard')" != 0 ]; then echo "[error] teleop is running"; exit 1; fi

# 視点: ロボットの左前・斜め 45°（4-12 節の後の静止画で決めた）
ign service -s /gui/move_to/pose --reqtype ignition.msgs.GUICamera --reptype ignition.msgs.Boolean --timeout 3000 \
  --req "pose: {position: {x: -0.21, y: -0.21, z: 1.21}, orientation: {x: 0, y: 0, z: 0.3826834, w: 0.9238795}}" > /dev/null
sleep 2

ffmpeg -hide_banner -loglevel error -y -f x11grab -framerate 15 -video_size $CROP_SIZE -i $CROP_AT \
  -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p "$OUT/raw.mp4" < /dev/null &
FF=$!
printf '%.3f\tvideo_start\n' "$(date +%s.%N)" >> "$EV"
sleep $TITLE_S

cd /home/ubuntu/pai_training
for t in "${TEXTS[@]}"; do
  printf '%.3f\ttext_start\t%s\n' "$(date +%s.%N)" "$t" >> "$EV"
  pixi run python /home/ubuntu/adv1/adv1_skill_runner.py --text "$t" --log-dir /home/ubuntu/adv1/logs --events "$EV" \
    >> "$OUT/run.log" 2>&1
  rc=$?
  printf '%.3f\ttext_end\t%s\t%s\n' "$(date +%s.%N)" "$t" "$rc" >> "$EV"
  echo "[text] $t rc=$rc"
  sleep $GAP_S
done

kill -INT $FF
wait $FF
printf '%.3f\tvideo_end\n' "$(date +%s.%N)" >> "$EV"
ffprobe -v error -show_entries format=duration -of default=nw=1 "$OUT/raw.mp4"
