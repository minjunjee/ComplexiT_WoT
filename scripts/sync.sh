#!/bin/bash
# Mac 터미널에서 실행한다(go.sh 를 돌리는 탭과 다른 탭에서):
#   bash scripts/sync.sh                      # https://github.com/minjunjee/ComplexiT_WoT 의 daehanwon/mind-complexity 브랜치
#   bash scripts/sync.sh <repo 주소> <브랜치>   # 다른 곳으로 올릴 때
#
# 하는 일 (Ctrl-C 로 언제든 멈춘다)
#   1. 처음 한 번: 이 폴더를 git 저장소로 만들고 원격의 main(이미 내 브랜치가 있으면 그 브랜치)에 이어 붙인다.
#      폴더의 파일은 바꾸지 않는다. 원격에만 있는 파일(CLAUDE.md)은 폴더로 받아 온다.
#   2. scripts/SYNC 의 번호가 새것이면: 바뀐 파일을 커밋하고(메시지는 scripts/COMMIT_MSG),
#      원격의 같은 브랜치에 새 커밋이 있으면 먼저 받아서 합친 뒤, 푸시한다.
# 지키는 것
#   - main 과 상대의 브랜치에는 푸시하지 않는다. force push 를 하지 않는다.
#   - .gitignore 에 적힌 것(서버 로그, 큰 출력, 접속 스크립트)은 올리지 않는다. CREDENTIAL 폴더는 이 저장소 밖에 있다.
#   - 푸시가 두 번 연속 실패하면 멈춘다.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); PROJ=$(dirname "$HERE")
URL=${1:-https://github.com/minjunjee/ComplexiT_WoT}
BR=${2:-daehanwon/mind-complexity}
LOGD="$PROJ/logs/bridge"; SYNC="$HERE/SYNC"; MSG="$HERE/COMMIT_MSG"; LAST="$LOGD/.last_sync"
mkdir -p "$LOGD"
cd "$PROJ" || exit 1
say() { echo "[sync] $(date '+%F %T') $*"; }
case "$BR" in daehanwon/*) ;; *) echo "브랜치 이름은 daehanwon/ 으로 시작해야 한다: $BR"; exit 1 ;; esac
{
  if [ ! -d .git ]; then
    git init -q && git symbolic-ref HEAD "refs/heads/$BR" && git remote add origin "$URL" || { say "git init 실패"; exit 1; }
  fi
  [ "$(git remote get-url origin 2>/dev/null)" = "$URL" ] || { say "이 폴더의 origin 이 $URL 이 아니다. 멈춘다."; exit 1; }
  git fetch -q origin || { say "원격을 읽지 못했다. 멈춘다."; exit 1; }
  if ! git rev-parse -q --verify HEAD >/dev/null; then
    BASE=origin/main; git rev-parse -q --verify "origin/$BR" >/dev/null && BASE="origin/$BR"
    git reset -q "$BASE" || { say "$BASE 에 이어 붙이지 못했다. 멈춘다."; exit 1; }
    git ls-files -d | while IFS= read -r f; do git checkout -q -- "$f"; done
    say "$BASE 에서 이어 간다. 브랜치: $BR"
  fi
  [ "$(git symbolic-ref --short HEAD 2>/dev/null)" = "$BR" ] || { say "지금 브랜치가 $BR 가 아니다. 멈춘다."; exit 1; }
  fails=0
  while true; do
    N=$(head -1 "$SYNC" 2>/dev/null)
    if [ -n "$N" ] && [ "$N" != "$(cat "$LAST" 2>/dev/null)" ]; then
      git add -A
      if git diff --cached --quiet; then say "$N: 새로 커밋할 것이 없다"
      elif git commit -q -F "$MSG"; then say "$N: 커밋 $(git rev-parse --short HEAD)"
      else say "$N: 커밋 실패"; fi
      git fetch -q origin
      if git rev-parse -q --verify "origin/$BR" >/dev/null && ! git merge-base --is-ancestor "origin/$BR" HEAD; then
        git pull -q --rebase origin "$BR" || { git rebase --abort 2>/dev/null; say "원격의 새 커밋과 합치지 못했다(충돌). 멈춘다."; echo "STOPPED" > "$LOGD/sync_heartbeat"; exit 1; }
        say "$N: 원격의 새 커밋을 받아 합쳤다"
      fi
      if git push -q -u origin "$BR"; then
        say "$N: 푸시 완료 $(git rev-parse --short HEAD) -> $URL ($BR)"; echo "$N" > "$LAST"; fails=0
      else
        fails=$((fails + 1)); say "$N: 푸시 실패 ($fails)"
        if [ "$fails" -ge 2 ]; then say "푸시가 두 번 연속 실패해서 멈춘다."; echo "STOPPED $(date '+%F %T')" > "$LOGD/sync_heartbeat"; exit 1; fi
      fi
    fi
    date '+%F %T' > "$LOGD/sync_heartbeat"
    sleep 30
  done
} 2>&1 | tee -a "$LOGD/sync.log"
