"""실험 4b: 가중치는 그대로 두고, 생성할 때 </think> 토큰의 로짓에 일정한 값(b)만 더한다.

실험 4에서 공개된 편집은 실제로 닫은 자리의 닫기 여유를 약 2.5 올렸다. 그렇다면 편집 없이 로짓에 그만큼만
더해도 편집한 모델처럼 생각을 닫고 같은 정답률이 나오는가를 본다. 나온다면 편집의 효과는 "닫기 문턱을 낮춘 것"
하나로 설명되고, 안 나온다면 편집이 그 밖의 일을 한다는 뜻이다.

  python -m cwot.exp4b_bias --weights base --bias 2.5 --n 48 --shard 0/4
  python -m cwot.exp4b_bias --weights mit  --bias 0   --n 48        # 확인: 이 생성 코드가 공개 기록(편집한 모델)을 재현하는가
출력: out/exp4b/<ds>_<weights>_b<bias>[_s<i>of<k>].jsonl (문제 하나에 한 줄, 이어서 돌 수 있다)
생성 설정은 공개 기록과 같다: temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

from . import common as C


class Presence(LogitsProcessor):
    """이미 생성한 토큰의 로짓에서 일정한 값을 뺀다(프롬프트의 토큰은 세지 않는다)."""

    def __init__(self, n_prompt, penalty):
        self.n_prompt, self.penalty = n_prompt, penalty

    def __call__(self, input_ids, scores):
        gen = input_ids[:, self.n_prompt:]
        if gen.shape[1]:
            pen = torch.zeros_like(scores)
            pen.scatter_(1, gen, self.penalty)
            scores = scores - pen
        return scores


class CloseBias(LogitsProcessor):
    """아직 생각을 닫지 않았을 때만 </think> 로짓에 b를 더한다."""

    def __init__(self, n_prompt, close_id, bias):
        self.n_prompt, self.close_id, self.bias = n_prompt, close_id, bias

    def __call__(self, input_ids, scores):
        if self.bias and not (input_ids[0, self.n_prompt:] == self.close_id).any():
            scores = scores.clone()
            scores[:, self.close_id] += self.bias
        return scores


class Stop(StoppingCriteria):
    """닫지 않은 채 생각이 max_think 를 넘으면 멈춘다. 닫은 뒤에는 max_answer 토큰까지만 쓴다."""

    def __init__(self, n_prompt, close_id, max_think, max_answer):
        self.n_prompt, self.close_id, self.max_think, self.max_answer = n_prompt, close_id, max_think, max_answer

    def __call__(self, input_ids, scores, **kw):
        gen = input_ids[0, self.n_prompt:]
        w = (gen == self.close_id).nonzero()
        if len(w):
            return bool(len(gen) - int(w[0]) - 1 >= self.max_answer)
        return bool(len(gen) >= self.max_think)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="gsm8k")
    ap.add_argument("--weights", default="base")
    ap.add_argument("--bias", type=float, default=2.5)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--max-think", type=int, default=16000)
    ap.add_argument("--max-answer", type=int, default=1500)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", default="")
    a = ap.parse_args()

    tok = C.load_tokenizer()
    model = C.load_model()
    ed = C.Editor(model)
    ed.set(a.weights)
    close_id = C.enc(tok, C.THINK_CLOSE)
    assert len(close_id) == 1
    close_id = close_id[0]
    probs = C.load_problems(a.ds)
    recs = C.load_records("p50c", a.ds)[:a.n]          # 문제 목록과 순서는 공개 기록과 같게 한다
    if a.shard:
        si, sk = map(int, a.shard.split("/"))
        recs = recs[si::sk]
    name = f"{a.ds}_{a.weights}_b{a.bias:g}" + (f"_{a.tag}" if a.tag else "") + (f"_s{a.shard.replace('/', 'of')}" if a.shard else "")
    os.makedirs(os.path.join(C.OUT, "exp4b"), exist_ok=True)
    path = os.path.join(C.OUT, "exp4b", name + ".jsonl")
    done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f"{name}: records={len(recs)} already={len(done)}", flush=True)
    t0 = time.time()
    with open(path, "a") as f:
        for i, r in enumerate(recs):
            if r["id"] in done:
                continue
            p = probs[r["id"]]
            ids = C.enc(tok, C.prompt_text(tok, p["problem"]))
            n0 = len(ids)
            torch.manual_seed(a.seed * 100003 + int(r["id"].split("_")[-1]) if r["id"].split("_")[-1].isdigit() else a.seed + i)
            x = torch.tensor([ids], device=model.device)
            with torch.no_grad():
                out = model.generate(
                    input_ids=x, attention_mask=torch.ones_like(x), do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                    repetition_penalty=1.0, max_new_tokens=a.max_think + a.max_answer, pad_token_id=pad,
                    logits_processor=LogitsProcessorList([Presence(n0, 1.5), CloseBias(n0, close_id, a.bias)]),
                    stopping_criteria=StoppingCriteriaList([Stop(n0, close_id, a.max_think, a.max_answer)]))
            gen = out[0, n0:].tolist()
            closed = close_id in gen
            n_think = gen.index(close_id) if closed else len(gen)
            ans_text = tok.decode(gen[n_think + 1:], skip_special_tokens=True) if closed else ""
            pred = C.last_boxed(ans_text)
            rec = {"id": r["id"], "weights": a.weights, "bias": a.bias, "closed": closed, "n_think": n_think, "n_gen": len(gen),
                   "pred": pred, "ok": C.same_answer(pred, p["answer"]), "gold": p["answer"],
                   "think_tail": tok.decode(gen[max(0, n_think - 60):n_think]), "answer_head": ans_text[:300],
                   "text": tok.decode(gen)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(recs)}] {r['id']} closed={closed} think={n_think} ok={rec['ok']} pred={pred} gold={p['answer']} "
                  f"{time.time() - t0:.0f}s", flush=True)
    rows = [json.loads(l) for l in open(path)]
    print(f"== {name}: n={len(rows)} closed={np.mean([x['closed'] for x in rows]):.3f} acc={np.mean([x['ok'] for x in rows]):.3f} "
          f"think median={np.median([x['n_think'] for x in rows]):.0f}")
    print("EXP4B DONE")


if __name__ == "__main__":
    main()
