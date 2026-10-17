#!/bin/bash
# コンテナ pai_ros2 の中で adv1_skill_runner.py を動かす（docker exec の bash -c は .bashrc を読まないので、ここで環境をそろえる）。
# 使い方: bash /home/ubuntu/adv1/adv1_run.sh "お辞儀して"
export PATH=/home/ubuntu/.pixi/bin:$PATH
source /opt/ros/humble/setup.bash
cd /home/ubuntu/pai_training
pixi run python /home/ubuntu/adv1/adv1_skill_runner.py --text "$1" --log-dir /home/ubuntu/adv1/logs 2>&1 | grep -v "WARN the lock file"
exit "${PIPESTATUS[0]}"
