"""실험 1. 끊어서 답 시켜 보기.

공개 풀이 기록의 생각을 여러 자리에서 끊고, 그 자리에서 곧바로 최종 답을 쓰게 한다(강제 답).
끊는 자리를 뒤로 옮겨도 강제 답이 더 바뀌지 않게 되는 가장 이른 자리가 '답이 정해진 지점'이다.

  python -m cwot.exp1_forced --run p50c --ds gsm8k --n 200
출력: out/exp1/<run>_<ds>.jsonl (기록 하나에 한 줄, 중간에 끊겨도 이어서 돈다)
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from . import common as C


def pick_cuts(ends, n_think, k):
    """끊을 자리: 0(생각 없음), 줄 끝 가운데 기하 간격으로 고른 자리들, 마지막(n_think)."""
    cand = np.array(sorted({e for e in ends if 0 < e <= n_think}))
    picks = {0, n_think}
    if len(cand):
        lo = max(8, int(cand[0]))
        if n_think > lo:
            for t in np.unique(np.round(np.geomspace(lo, n_think, k)).astype(int)):
                i = int(np.searchsorted(cand, t, side="right")) - 1
                picks.add(int(cand[max(i, 0)]))
    return sorted(picks)


@torch.no_grad()
def forced_answer(model, tok, ids, max_new):
    x = torch.tensor([ids], device=model.device)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=max_new,
                         do_sample=False, pad_token_id=pad)
    text = tok.decode(out[0, len(ids):], skip_special_tokens=True)
    return C.close_brace(text), text


def stable_from(answers):
    """마지막 강제 답과 같은 답이 끝까지 이어지기 시작하는 가장 이른 위치(인덱스). 마지막 답이 없으면 None."""
    last = C.norm_answer(answers[-1])
    if last is None:
        return None
    j = len(answers) - 1
    while j > 0 and C.norm_answer(answers[j - 1]) == last:
        j -= 1
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="p50c", choices=["p50c", "unedited"])
    ap.add_argument("--ds", default="gsm8k", choices=["gsm8k", "math500"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--weights", default="auto", help="auto: p50c 기록은 mit, unedited 기록은 base")
    ap.add_argument("--cuts", type=int, default=20)
    ap.add_argument("--max-think", type=int, default=16000, help="생각 토큰 수 상한(넘으면 앞부분만 쓴다)")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--mismatch", action="store_true", help="대조: 다른 문제의 생각을 넣는다")
    ap.add_argument("--tag", default="")
    ap.add_argument("--shard", default="", help="i/k: 기록 i, i+k, i+2k ... 만 돌린다(한 GPU에 여러 프로세스를 올릴 때)")
    a = ap.parse_args()

    weights = a.weights if a.weights != "auto" else ("mit" if a.run == "p50c" else "base")
    tok = C.load_tokenizer()
    model = C.load_model()
    C.Editor(model).set(weights)
    probs = C.load_problems(a.ds)
    recs = C.load_records(a.run, a.ds)[:a.n]
    all_recs = recs
    if a.shard:
        si, sk = map(int, a.shard.split("/"))
        recs = recs[si::sk]
    cue = C.enc(tok, C.CUE)

    os.makedirs(os.path.join(C.OUT, "exp1"), exist_ok=True)
    name = (f"{a.run}_{a.ds}" + ("_mismatch" if a.mismatch else "") + (f"_{a.tag}" if a.tag else "")
            + (f"_s{a.shard.replace('/', 'of')}" if a.shard else ""))
    path = os.path.join(C.OUT, "exp1", name + ".jsonl")
    done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()
    print(f"{name}: weights={weights} records={len(recs)} already={len(done)}", flush=True)

    t0 = time.time()
    with open(path, "a") as f:
        for i, r in enumerate(recs):
            if r["id"] in done:
                continue
            gi = all_recs.index(r)
            src = all_recs[(gi + 1) % len(all_recs)] if a.mismatch else r      # 생각을 가져올 기록
            think, ans_text, closed = C.split_thinking(src["text"])
            p_ids = C.enc(tok, C.prompt_text(tok, probs[r["id"]]["problem"]))
            t_all = C.enc(tok, think)
            t_ids = t_all[:a.max_think]
            cuts = pick_cuts(C.line_ends(tok, t_ids), len(t_ids), a.cuts)
            if a.mismatch:
                cuts = cuts[-4:]
            gold = probs[r["id"]]["answer"]
            final = C.last_boxed(C.split_thinking(r["text"])[1]) if C.THINK_CLOSE in r["text"] else None
            rows = []
            for c in cuts:
                ans, raw = forced_answer(model, tok, p_ids + t_ids[:c] + cue, a.max_new)
                rows.append({"cut": c, "answer": ans, "raw": raw[:80], "ok_gold": C.same_answer(ans, gold),
                             "eq_final": C.norm_answer(ans) is not None and C.norm_answer(ans) == C.norm_answer(final)})
            j = stable_from([x["answer"] for x in rows])
            rec = {"id": r["id"], "run": a.run, "ds": a.ds, "weights": weights, "mismatch": a.mismatch,
                   "think_src": src["id"], "n_think": len(t_all), "n_think_used": len(t_ids), "closed": closed,
                   "gold": gold, "final": final, "correct_first": r["correct_first"], "cuts": rows,
                   "settled_cut": None if j is None else rows[j]["cut"],
                   "settled_frac": None if j is None or not len(t_ids) else rows[j]["cut"] / len(t_ids)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 2 == 0 or i == len(recs) - 1:
                print(f"[{i + 1}/{len(recs)}] {r['id']} think={len(t_all)} cuts={len(cuts)} settled={rec['settled_cut']} "
                      f"ok0={rows[0]['ok_gold']} okL={rows[-1]['ok_gold']} {time.time() - t0:.0f}s", flush=True)
    summarize(path)


def summarize(path):
    rows = [json.loads(l) for l in open(path)]
    if not rows:
        return
    sf = [r["settled_frac"] for r in rows if r["settled_frac"] is not None]
    print(f"\n== {os.path.basename(path)}: {len(rows)} records")
    print("강제 답이 안정된 기록:", len(sf), "| 답이 정해진 지점(생각 길이 대비) 중앙값:",
          round(float(np.median(sf)), 3) if sf else None,
          "| 그 뒤에 쓴 생각의 비율 중앙값:", round(1 - float(np.median(sf)), 3) if sf else None)
    print("생각 없이(cut 0) 정답:", round(np.mean([r["cuts"][0]["ok_gold"] for r in rows]), 3),
          "| 마지막 자리에서 정답:", round(np.mean([r["cuts"][-1]["ok_gold"] for r in rows]), 3),
          "| 공개 기록의 첫 응답 정답:", round(np.mean([r["correct_first"] for r in rows]), 3))


if __name__ == "__main__":
    main()
