# ComplexiT_WoT

추론 모델의 mind 복잡도를 가중치 부품 단위로 재고, 답에 필요한 계산과 필요하지 않은 계산을 가르고,
필요하지 않은 것만 학습 없이 치우는 연구(후보 1)의 실험 코드다.

- 연구 계획: [후보 1 재설계안](https://claude.ai/artifact/8EESDJcf1moq5oDsSRWrbS)
- 지금까지의 결과: `results/RESULTS.md`
- 지금 상태와 다음에 할 일: `HANDOFF.md`
- 시도한 것과 결과의 기록: `NOTES.md`

## 구조

| 경로 | 내용 |
|---|---|
| `cwot/common.py` | 모델 로드, MIT 편집 적용(해시 대조), 공개 풀이 기록 로드, 프롬프트 재구성, 답 추출, 닫기 여유 계산 |
| `cwot/prep.py` | 모델과 문제 데이터를 미리 받는다 |
| `cwot/smoke.py` | 실험 전 확인 9가지 |
| `cwot/exp1_forced.py` | 실험 1. 생각을 끊고 답을 시켜서 답이 정해진 지점을 찾는다 |
| `cwot/exp4_margin.py` | 실험 4. 가중치 변형별 닫기 여유 곡선 |
| `cwot/exp4b_bias.py` | 실험 4b. 가중치는 그대로 두고 `</think>` 로짓에 일정한 값만 더해 생성한다 |
| `cwot/ana.py` | 실험 1·4의 요약(서버에서 돈다) |
| `scripts/make_report.py` | 요약과 원자료로 표와 그림을 만든다(`results/`) |
| `results/` | 결과 정리, 표, 그림, 기록별 원자료 |
| `remote/` | cs103에서 도는 단계 스크립트(`pipeline.sh`)와 환경(`env.sh`) |
| `scripts/go.sh` | Mac에서 실행하는 다리. 코드를 올리고, 단계를 띄우고, 로그를 가져온다. 서버 주소가 들어 있어 저장소에는 올리지 않는다 |
| `scripts/sync.sh` | Mac에서 실행한다. `scripts/SYNC`의 번호가 바뀌면 커밋하고 GitHub에 푸시한다 |
| `scripts/REQUEST` | `"<번호> <단계>"` 한 줄. 번호가 바뀌면 `go.sh`가 그 단계를 띄운다 |
| `remote_out/` | cs103에서 가져온 로그와 결과 (git에 넣지 않는다) |

## 실행

Mac 터미널에서 한 줄을 실행하고 켜 둔다.

```sh
bash ~/KAIST/courses/2026F/AIC330_llm/Introduction_LLM/ComplexiT_WoT/scripts/go.sh
```

단계 이름은 `remote/pipeline.sh`에 있다: `check`, `setup`, `smoke`, `pilot`, `first`(앞의 넷을 차례로), `fast`(실험 1·4를 기록 조각으로 나눠 여러 프로세스로), `ana`, `bias`(실험 4b), `exp1`, `exp4`, `week1`(exp1과 exp4를 GPU 0과 1에서 동시에).

## 규칙

- cs103에서는 GPU 물리 번호 0, 1만 쓴다. 사용 중이면 올리지 않는다. 4–7번은 건드리지 않는다.
- cs103의 작업 경로는 `/data/daehan/Introduction_LLM/ComplexiT_WoT` 아래로 한정한다. 공용 venv(`/data/daehan/venv`)는 고치지 않는다.
- 비밀번호 파일(`CREDENTIAL/DONOTREAD/`)은 열지 않는다. `sshpass -f`로만 넘긴다.
- 실험은 중간에 끊겨도 같은 명령으로 이어서 돌 수 있게 만든다.
