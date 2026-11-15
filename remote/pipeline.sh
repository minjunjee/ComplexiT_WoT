#!/bin/bash
# cs103에서 도는 실험 단계. 사용: pipeline.sh <stage> [nonce]
# 로그: $CWOT_ROOT/logs/<nonce>_<stage>.log, 진행 상태: $CWOT_ROOT/logs/STATUS
set -u
. "$(dirname "$0")/env.sh"
STAGE=${1:?stage}; NONCE=${2:-manual}
mkdir -p "$CWOT_ROOT/logs" "$CWOT_ROOT/out" "$CWOT_ROOT/data" "$TMPDIR" "$HF_HOME"
cd "$CWOT_ROOT"
LOG=$CWOT_ROOT/logs/${NONCE}_${STAGE}.log
status() { echo "$(date -u +%FT%TZ) $NONCE $STAGE $*" >> "$CWOT_ROOT/logs/STATUS"; }

gpu_guard() {   # 인자로 받은 GPU가 비어 있는지 본다. 다른 작업이 있으면 올리지 않는다.
  for g in "$@"; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')
    if [ "${used:-99999}" -gt 2000 ]; then echo "GPU $g 는 사용 중이다(${used} MiB). 올리지 않는다."; return 1; fi
  done
}

st_check() {
  echo "== 시각"; date -u; TZ=Asia/Seoul date; hostname; whoami
  echo "== GPU"; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
  echo "== 디스크"; df -h /data | tail -1; du -sh /data/daehan 2>/dev/null
  echo "== 파이썬"; python3 --version; command -v uv conda mamba virtualenv; echo "PY=$PY"
  $PY -c "import sys, torch; print(sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | tail -1
  echo "== /data/daehan"; ls /data/daehan | head -40
  for m in transformers datasets accelerate safetensors huggingface_hub math_verify; do
    $PY -c "import $m; print('$m', getattr($m, '__version__', '?'))" 2>&1 | tail -1
  done
  echo "== 네트워크"
  curl -sS -m 10 -o /dev/null -w "github %{http_code}\n" https://github.com 2>&1 | tail -1
  curl -sS -m 10 -o /dev/null -w "huggingface %{http_code}\n" https://huggingface.co/api/models/Qwen/Qwen3.5-0.8B 2>&1 | tail -1
  echo "== 기타"; tmux -V; ls "$HF_HOME/hub" 2>/dev/null | head -20
}

st_setup() {
  # 프로젝트 전용 venv를 /data 아래에 만든다. 다른 venv는 건드리지 않는다.
  # (2026-09 문서의 /data/daehan/venv 는 지금 없다. 2026-11-16 확인)
  V=$CWOT_ROOT/venv
  if [ ! -f "$V/.ready" ]; then
    rm -rf "$V"
    MODE=""
    UV=$(command -v uv 2>/dev/null || true); [ -n "$UV" ] || { [ -x "$HOME/.local/bin/uv" ] && UV=$HOME/.local/bin/uv; }
    if python3 -m venv "$V" >/dev/null 2>&1 && "$V/bin/python" -m pip --version >/dev/null 2>&1; then MODE=pip
    else
      # 이 서버의 python3 에는 ensurepip 이 없다. pip 없이 venv를 만들고 get-pip.py 로 pip을 넣는다.
      rm -rf "$V"
      if python3 -m venv --without-pip "$V" >/dev/null 2>&1 && curl -sS -m 60 https://bootstrap.pypa.io/get-pip.py -o "$TMPDIR/get-pip.py" \
         && "$V/bin/python" "$TMPDIR/get-pip.py" -q 2>&1 | tail -2 && "$V/bin/python" -m pip --version >/dev/null 2>&1; then MODE=pip
      else
        rm -rf "$V"
        if [ -n "$UV" ] && "$UV" venv "$V" >/dev/null 2>&1 && [ -x "$V/bin/python" ]; then MODE=uv; fi
      fi
    fi
    [ -n "$MODE" ] || { echo "venv를 만들지 못했다(세 가지 방법 모두 실패)"; python3 -m venv --without-pip "$V" 2>&1 | tail -3; return 1; }
    echo "venv 방식: $MODE"
    pipi() { if [ "$MODE" = uv ]; then "$UV" pip install --python "$V/bin/python" "$@"; else "$V/bin/python" -m pip install -q "$@"; fi; }
    pipi torch 2>&1 | tail -3 || return 1
    pipi "transformers>=5.17" datasets safetensors huggingface_hub math-verify accelerate 2>&1 | tail -3 || return 1
    touch "$V/.ready"
  fi
  PY=$V/bin/python
  $PY -c "import torch, transformers, numpy; print('torch', torch.__version__, 'transformers', transformers.__version__, 'numpy', numpy.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())" || return 1
  if [ ! -d "$CWOT_ROOT/data/mit/.git" ]; then
    git clone --depth 1 https://github.com/RikiyaT/lm_understand-logs "$CWOT_ROOT/data/mit" 2>&1 | tail -2 || return 1
  fi
  $PY -m cwot.prep
}

st_smoke() { gpu_guard 0 && CUDA_VISIBLE_DEVICES=0 $PY -m cwot.smoke; }

st_pilot() {
  gpu_guard 0 1 || return 1
  CUDA_VISIBLE_DEVICES=0 $PY -m cwot.exp1_forced --run p50c --ds gsm8k --n 6 --tag pilot > "$CWOT_ROOT/logs/${NONCE}_pilot_exp1.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=1 $PY -m cwot.exp4_margin --run p50c --ds gsm8k --n 6 > "$CWOT_ROOT/logs/${NONCE}_pilot_exp4.log" 2>&1 &
  wait
  tail -8 "$CWOT_ROOT/logs/${NONCE}_pilot_exp1.log"; tail -4 "$CWOT_ROOT/logs/${NONCE}_pilot_exp4.log"
}

st_exp1() {   # GPU 0: 편집한 모델의 기록부터. 중간에 끊겨도 같은 명령으로 이어서 돈다.
  gpu_guard 0 || return 1
  for spec in "p50c gsm8k" "p50c math500" "unedited gsm8k" "unedited math500"; do
    set -- $spec
    CUDA_VISIBLE_DEVICES=0 $PY -m cwot.exp1_forced --run "$1" --ds "$2" --n 200 || return 1
  done
  CUDA_VISIBLE_DEVICES=0 $PY -m cwot.exp1_forced --run p50c --ds gsm8k --n 100 --mismatch
}

st_exp4() {   # GPU 1
  gpu_guard 1 || return 1
  for spec in "p50c gsm8k" "p50c math500" "unedited gsm8k" "unedited math500"; do
    set -- $spec
    CUDA_VISIBLE_DEVICES=1 $PY -m cwot.exp4_margin --run "$1" --ds "$2" --n 200 || return 1
  done
}

par() {   # par <exp1|exp4> <gpu> <run> <ds> <n> <조각 수> [추가 인자]: 기록을 조각으로 나눠 한 GPU에 여러 프로세스로 올리고 끝날 때까지 기다린다.
  local exp=$1 gpu=$2 run=$3 ds=$4 n=$5 k=$6; shift 6
  local mod=cwot.exp1_forced; [ "$exp" = exp4 ] && mod=cwot.exp4_margin
  local pids="" i
  for i in $(seq 0 $((k - 1))); do
    CUDA_VISIBLE_DEVICES=$gpu $PY -m $mod --run "$run" --ds "$ds" --n "$n" --shard $i/$k "$@" >> "$CWOT_ROOT/logs/${NONCE}_${exp}_${run}_${ds}${1:+_x}_s$i.log" 2>&1 &
    pids="$pids $!"
  done
  wait $pids
  echo "$(date -u +%T) done: $exp $run $ds n=$n $*"
}

st_fast() {   # 중간 보고용. 편집한 모델의 GSM8K 기록이 가장 먼저 끝나도록 순서를 잡았다. 끊겨도 같은 명령으로 이어서 돈다.
  gpu_guard 0 1 || return 1
  ( par exp1 0 p50c gsm8k 200 5; par exp1 0 p50c math500 200 5; par exp1 0 p50c gsm8k 100 5 --mismatch ) &
  local c1=$!
  ( par exp4 1 p50c gsm8k 200 3; par exp4 1 unedited gsm8k 200 3; par exp4 1 p50c math500 200 3; par exp4 1 unedited math500 200 3 ) &
  local c2=$!
  ( par exp1 1 unedited gsm8k 200 4; par exp1 1 unedited math500 200 4 ) &
  local c3=$!
  # 도는 동안 3분마다 요약을 새로 만든다. 끝나면 한 번 더.
  ( while true; do sleep 180; $PY -m cwot.ana > "$CWOT_ROOT/logs/${NONCE}_ana.tmp" 2>&1; mv "$CWOT_ROOT/logs/${NONCE}_ana.tmp" "$CWOT_ROOT/logs/${NONCE}_ana.log"; done ) &
  local loop=$!
  wait $c1 $c2 $c3
  kill $loop 2>/dev/null
  $PY -m cwot.ana > "$CWOT_ROOT/logs/${NONCE}_ana.log" 2>&1
  echo FAST DONE
}

gpu_guard_own() {   # 그 GPU에 올라가 있는 프로세스가 모두 이 프로젝트의 것(cwot.*)일 때만 통과한다.
  local g pid
  for g in "$@"; do
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" | tr -d ' '); do
      tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -q "cwot\." || { echo "GPU $g 에 다른 작업(pid $pid)이 있다. 올리지 않는다."; return 1; }
    done
  done
}

st_bias() {   # 실험 4b: 로짓에 일정한 값만 더해 생성. 조건 둘(원래 가중치 + 2.5, 편집한 가중치 + 0), 조건마다 조각 8개.
  gpu_guard_own 0 1 || return 1
  local pids="" i g
  for i in 0 1 2 3 4 5 6 7; do
    g=$((i % 2))
    CUDA_VISIBLE_DEVICES=$g $PY -m cwot.exp4b_bias --weights base --bias 2.5 --n 48 --tag v2 --shard $i/8 >> "$CWOT_ROOT/logs/${NONCE}_b2.5_s$i.log" 2>&1 & pids="$pids $!"
    CUDA_VISIBLE_DEVICES=$g $PY -m cwot.exp4b_bias --weights mit --bias 0 --n 48 --tag v2 --shard $i/8 >> "$CWOT_ROOT/logs/${NONCE}_mit_s$i.log" 2>&1 & pids="$pids $!"
  done
  wait $pids
  tail -n 2 "$CWOT_ROOT"/logs/${NONCE}_*_s*.log
}

st_rebias() {   # 앞서 띄운 실험 4b(tmux 세션 cwot_r006)를 내리고 설정을 고쳐 다시 띄운다. 내리는 대상은 그 세션 하나뿐이다.
  tmux kill-session -t cwot_r006 2>/dev/null; sleep 6
  local pid
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0,1 | tr -d ' '); do
    # 세션을 내려도 남은 것이 있으면, 이 프로젝트 폴더에서 돈 exp4b 프로세스인지 확인한 뒤 그 PID만 내린다.
    if tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -q "cwot\.exp4b_bias" && [ "$(readlink /proc/$pid/cwd)" = "$CWOT_ROOT" ] \
       && ! tr '\0' ' ' < /proc/$pid/cmdline | grep -q -- "--tag v2"; then echo "남은 프로세스 $pid 를 내린다"; kill "$pid"; fi
  done
  sleep 3
  st_bias
}

run_stage() {
  case "$STAGE" in
    check) st_check ;;
    setup) st_setup ;;
    smoke) st_smoke ;;
    pilot) st_pilot ;;
    first) st_check && st_setup && st_smoke && st_pilot ;;
    ana) $PY -m cwot.ana ;;
    fast) st_fast ;;
    bias) st_bias ;;
    biascheck) gpu_guard_own 0 && CUDA_VISIBLE_DEVICES=0 $PY -m cwot.biascheck ;;
    rebias) st_rebias ;;
    exp1) st_exp1 ;;
    exp4) st_exp4 ;;
    week1) gpu_guard 0 1 || return 1; ( st_exp1 > "$CWOT_ROOT/logs/${NONCE}_exp1.log" 2>&1 & st_exp4 > "$CWOT_ROOT/logs/${NONCE}_exp4.log" 2>&1 & wait ) ;;
    *) echo "모르는 단계: $STAGE"; return 2 ;;
  esac
}

status START
( run_stage ) > "$LOG" 2>&1
rc=$?
status "END rc=$rc"
exit $rc
