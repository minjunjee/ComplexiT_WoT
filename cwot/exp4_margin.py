"""실험 4. 닫기 여유 곡선.

같은 풀이 글을 가중치만 바꿔 읽히고, 자리마다 닫기 여유(</think> 점수 - 나머지 최고 점수)를 기록한다.
margin[v, j]는 '글 토큰 j개 뒤에 </think>를 낼 것인가'에 해당하고, 실험 1의 cut j와 같은 자리다.

  python -m cwot.exp4_margin --run p50c --ds gsm8k --n 200 --variants base,a0.5,a1,a2,a3,mit
출력: out/exp4/<run>_<ds>/<id>.npz (있으면 건너뛴다)
"""
import argparse
import json
import os
import re
import time

import numpy as np
import torch

from . import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="p50c", choices=["p50c", "unedited"])
    ap.add_argument("--ds", default="gsm8k", choices=["gsm8k", "math500"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--variants", default="base,a0.5,a1,a2,a3,mit")
    ap.add_argument("--max-tokens", type=int, default=16000, help="글 토큰 수 상한")
    ap.add_argument("--shard", default="", help="i/k: 기록 i, i+k, i+2k ... 만 돌린다")
    a = ap.parse_args()

    variants = a.variants.split(",")
    tok = C.load_tokenizer()
    model = C.load_model()
    ed = C.Editor(model)
    close_id = C.enc(tok, C.THINK_CLOSE)
    assert len(close_id) == 1, close_id
    close_id = close_id[0]
    probs = C.load_problems(a.ds)
    recs = C.load_records(a.run, a.ds)[:a.n]
    if a.shard:
        si, sk = map(int, a.shard.split("/"))
        recs = recs[si::sk]
    out_dir = os.path.join(C.OUT, "exp4", f"{a.run}_{a.ds}")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    for i, r in enumerate(recs):
        path = os.path.join(out_dir, re.sub(r"[^A-Za-z0-9_.-]", "_", r["id"]) + ".npz")
        if os.path.exists(path):
            continue
        p_ids = C.enc(tok, C.prompt_text(tok, probs[r["id"]]["problem"]))
        x_all = C.enc(tok, r["text"])
        x_ids = x_all[:a.max_tokens]
        ids = torch.tensor([p_ids + x_ids], device=model.device)
        tg = torch.tensor(x_ids, device=model.device)
        margin, logp, nll = [], [], []
        for v in variants:
            ed.set(v)
            h = C.hidden_last(model, ids)[len(p_ids) - 1:]
            m, lp, act = C.close_margin(model, h, close_id, targets=tg)
            margin.append(m.cpu().numpy().astype(np.float32))
            logp.append(lp.cpu().numpy().astype(np.float32))
            nll.append((-act).cpu().numpy().astype(np.float32))
        ed.set("base")
        close_pos = x_ids.index(close_id) if close_id in x_ids else -1
        np.savez_compressed(path, margin=np.stack(margin), logp_close=np.stack(logp), nll=np.stack(nll),
                            line_ends=np.array(C.line_ends(tok, x_ids), dtype=np.int32),
                            meta=json.dumps({"id": r["id"], "run": a.run, "ds": a.ds, "variants": variants,
                                             "n_prompt": len(p_ids), "n_text": len(x_all), "n_used": len(x_ids),
                                             "close_pos": close_pos, "correct_first": r["correct_first"],
                                             "gold": probs[r["id"]]["answer"]}))
        if (i + 1) % 2 == 0 or i == len(recs) - 1:
            mm = np.stack(margin)
            k = close_pos if close_pos >= 0 else mm.shape[1] - 1
            print(f"[{i + 1}/{len(recs)}] {r['id']} n={len(x_ids)} close_pos={close_pos} "
                  + " ".join(f"{v}:{mm[q, :max(k, 1)].mean():.1f}" for q, v in enumerate(variants))
                  + f" | {time.time() - t0:.0f}s", flush=True)
    print("EXP4 DONE", flush=True)


if __name__ == "__main__":
    main()
