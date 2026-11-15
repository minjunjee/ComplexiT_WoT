#!/bin/bash
# 단계를 tmux(없으면 nohup)로 띄우고 바로 돌아온다. 사용: launch.sh <nonce> <stage>
set -u
. "$(dirname "$0")/env.sh"
NONCE=${1:?nonce}; STAGE=${2:?stage}
mkdir -p "$CWOT_ROOT/logs"
if command -v tmux >/dev/null 2>&1; then
  tmux has-session -t "cwot_$NONCE" 2>/dev/null && { echo "이미 떠 있다: cwot_$NONCE"; exit 0; }
  tmux new-session -d -s "cwot_$NONCE" "bash $CWOT_ROOT/code/remote/pipeline.sh $STAGE $NONCE"
  echo "tmux cwot_$NONCE 로 $STAGE 를 띄웠다"
else
  nohup setsid bash "$CWOT_ROOT/code/remote/pipeline.sh" "$STAGE" "$NONCE" > /dev/null 2>&1 &
  echo "nohup 으로 $STAGE 를 띄웠다"
fi
