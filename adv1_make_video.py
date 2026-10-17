#!/usr/bin/env python3
"""段 4: adv1_record_demo.sh の録画（raw.mp4）と時刻の記録（events.tsv）から字幕を作り、動画に焼き込む。

字幕（10/12 に決めたもの）:
  冒頭: 題名（omnicampus のアカウント名を含む。要綱 10/8 版で記載可）
  上  : 指示の文
  下  : 実行中の動き（何番目か）／直立に戻る／知らない言葉で動かない
使い方（コンテナの中。標準の python3 で動く）:
    python3 /home/ubuntu/adv1/adv1_make_video.py /home/ubuntu/adv1/video/take1 adv1_demo_take1.mp4
"""
import os
import subprocess
import sys

NAMES = {'pekori': 'お辞儀', 'unazuki': 'うなずく', 'furu': '手を振る', 'paku': 'ぱく（つかむ）'}
TITLE = ('言葉の指示で、模倣学習した動きを組み合わせる\\N'
         '{\\fs20}CRANE+ V2（Gazebo）・ACT（模倣学習）× 4 つの動き\\N'
         '{\\fs20}PAI最終課題_ogawa_one')  # 要綱 10/8 版の YouTube のタイトルの書式「PAI最終課題_omnicampusアカウント名」にそろえる
W, H = 580, 620
FONT = 'Noto Sans CJK JP'

src = sys.argv[1]
out_name = sys.argv[2]
rows = [line.rstrip('\n').split('\t') for line in open(os.path.join(src, 'events.tsv')) if line.strip()]
t0 = next(float(r[0]) for r in rows if r[1] == 'video_start')
t_end = next(float(r[0]) for r in rows if r[1] == 'video_end')


def ts(t):
    t = max(0.0, t - t0)
    return f'{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}'


events = []  # (start, end, style, text)
first_text = next(float(r[0]) for r in rows if r[1] == 'text_start')
events.append((t0, first_text, 'Title', TITLE))

text_starts = [(float(r[0]), r[2]) for r in rows if r[1] == 'text_start']
text_ends = {r[2]: (float(r[0]), r[3]) for r in rows if r[1] == 'text_end'}
for k, (start, text) in enumerate(text_starts):
    nxt = text_starts[k + 1][0] if k + 1 < len(text_starts) else t_end
    events.append((start, nxt, 'Top', f'指示:「{text}」'))
    end, rc = text_ends[text]
    if rc == '2':
        # 言葉を読み取った時刻から出す（それまでは「プログラムを起動中」）
        parsed = next((float(r[0]) for r in rows if r[1] == 'parse' and r[2] == text), start)
        events.append((parsed, nxt, 'Bottom', '知っている動きの言葉が無いので、動きません'))

current = None
last_text_start = None
for r in rows:
    t, kind = float(r[0]), r[1]
    if kind == 'text_start':
        last_text_start = t
    elif kind == 'parse' and last_text_start is not None:
        # 指示を出してから言葉を読み取るまでは、プログラム（pixi・Python）の起動の時間
        events.append((last_text_start, t, 'Bottom', '（プログラムを起動中）'))
    if kind == 'parse' and r[3]:
        seq = [NAMES.get(s, s) for s in r[3].split(',')]
        events.append((t, t + 2.0, 'Bottom', '→ 動きの並び: ' + ' → '.join(seq)))
        # 並びを出した後、最初の動きが始まるまではモデルの読み込み（次の skill_start で詰められる）
        events.append((t + 2.0, t + 60.0, 'Bottom', '（モデルを読み込み中）'))
    elif kind == 'skill_start':
        current = (t, f'実行中: {NAMES.get(r[4], r[4])}（{r[2]}/{r[3]}）')
    elif kind == 'skill_end' and current:
        events.append((current[0], t, 'Bottom', current[1]))
        current = (t, '直立に戻る')
    elif kind == 'upright_end' and current:
        events.append((current[0], t, 'Bottom', current[1]))
        current = None

ass = os.path.join(src, 'subs.ass')
with open(ass, 'w') as f:
    f.write(f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Title,{FONT},26,&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,1,0,0,0,100,100,0,0,3,10,0,5,20,20,20,1
Style: Top,{FONT},24,&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,1,0,0,0,100,100,0,0,3,8,0,8,10,10,14,1
Style: Bottom,{FONT},22,&H0000FFFF,&H00FFFFFF,&H00000000,&H96000000,1,0,0,0,100,100,0,0,3,8,0,2,10,10,14,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""")
    # 同じ場所に重なる字幕は、後から始まるほうを優先して前のものの終わりを詰める
    events.sort(key=lambda e: (e[2], e[0]))
    for i, (s, e, style, text) in enumerate(events):
        nxt = next((x[0] for x in events[i + 1:] if x[2] == style and x[0] > s), None)
        if nxt is not None and nxt < e:
            e = nxt
        if e - s < 0.5:  # すでに直立のときの「直立に戻る」等、一瞬だけの字幕は出さない（チラつく）
            continue
        f.write(f'Dialogue: 0,{ts(s)},{ts(e)},{style},,0,0,0,,{text}\n')
print(f'[subs] {ass} lines={len(events)}')

out = os.path.join(src, out_name)
cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', os.path.join(src, 'raw.mp4'),
       '-vf', f'subtitles={ass}:fontsdir=/usr/share/fonts/opentype/noto',
       '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out]
subprocess.run(cmd, check=True)
print(f'[video] {out}')
