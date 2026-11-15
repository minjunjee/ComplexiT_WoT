"""실험 전에 확인할 것을 한 번에 본다. 각 항목은 실패해도 다음 항목으로 넘어간다.

1. 버전과 GPU   2. 토크나이저: </think> 토큰, 프롬프트 끝 모양   3. 모델 클래스와 파라미터 이름
4. MIT 편집 적용과 해시 대조, delta의 구조(열 수, rank-1 정도, 추정 배수)
5. 본체 hidden + 출력 행렬 = 모델 전체 로짓인지   6. 왼쪽 패딩이 결과를 바꾸는지
7. 공개 풀이 글이 이 프롬프트에서 나온 것이 맞는지(NLL)   8. 짧은 생성   9. 순전파 시간
"""
import json
import time
import traceback

import torch

from . import common as C


def step(name):
    print(f"\n===== {name} =====", flush=True)


def main():
    step("1. versions / GPU")
    import transformers
    print("torch", torch.__version__, "transformers", transformers.__version__)
    print("cuda", torch.cuda.is_available(), "count", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i), round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1), "GiB")

    step("2. tokenizer")
    tok = C.load_tokenizer()
    close_ids = C.enc(tok, C.THINK_CLOSE)
    print("</think> ids:", close_ids, "| single token:", len(close_ids) == 1)
    probs = C.load_problems("gsm8k")
    p0 = probs["gsm8k_test_0"]["problem"]
    pt = C.prompt_text(tok, p0)
    print("prompt tail (thinking on):", repr(pt[-80:]))
    print("prompt tail (thinking off):", repr(C.prompt_text(tok, p0, thinking=False)[-80:]))
    print("cue ids:", C.enc(tok, C.CUE))
    print("pad:", tok.pad_token_id, "eos:", tok.eos_token_id, "vocab:", len(tok))

    step("3. model")
    model = C.load_model()
    print("class:", type(model).__name__, "| base_model:", type(model.base_model).__name__)
    print("params:", sum(p.numel() for p in model.parameters()) / 1e6, "M | dtype:", next(model.parameters()).dtype)
    names = [n for n, _ in model.named_parameters()]
    print("n tensors:", len(names))
    for n in names:
        if ".layers.15." in n or n.endswith("layers.15.mlp.down_proj.weight"):
            print("  ", n, tuple(dict(model.named_parameters())[n].shape))
    print("first 5:", names[:5], "| last 3:", names[-3:])

    ed = None
    try:
        step("4. MIT edit")
        ed = C.Editor(model, strict=False)
        for d in ed.describe():
            print(json.dumps(d))
        for name, u in ed.directions().items():
            print("direction", name, "norm", float(u.norm()))
        us = list(ed.directions().values())
        if len(us) == 2:
            print("cos(u15, u19) =", float(torch.dot(us[0], us[1])))
    except Exception:
        traceback.print_exc()

    close_id = close_ids[0]
    recs = C.load_records("p50c", "gsm8k")
    r = next(x for x in recs if C.THINK_CLOSE in x["text"])
    p_ids = C.enc(tok, C.prompt_text(tok, probs[r["id"]]["problem"]))
    x_ids = C.enc(tok, r["text"])[:1500]
    ids = torch.tensor([p_ids + x_ids], device=model.device)

    try:
        step("5. base_model + head == full logits ?")
        short = ids[:, :300]
        full = model(input_ids=short).logits[0].float()
        h = C.hidden_last(model, short)
        mine = model.get_output_embeddings()(h).float()
        print("max abs diff:", float((full - mine).abs().max()), "| argmax agree:", float((full.argmax(-1) == mine.argmax(-1)).float().mean()))
    except Exception:
        traceback.print_exc()

    try:
        step("6. left padding changes the last-position logits ?")
        short = ids[:, :200]
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        padded = torch.cat([torch.full((1, 7), pad, device=model.device), short], dim=1)
        mask = torch.cat([torch.zeros((1, 7), dtype=torch.long, device=model.device), torch.ones_like(short)], dim=1)
        a = model(input_ids=short).logits[0, -1].float()
        b = model(input_ids=padded, attention_mask=mask).logits[0, -1].float()
        print("max abs diff:", float((a - b).abs().max()), "| same argmax:", bool(a.argmax() == b.argmax()))
        print("-> 차이가 크면 배치(패딩) 생성을 쓰면 안 된다. exp1은 지금 한 번에 하나씩만 돌린다.")
    except Exception:
        traceback.print_exc()

    try:
        step("7. teacher-forced NLL of a public p50c transcript (first 1500 tokens)")
        tg = torch.tensor(x_ids, device=model.device)
        for v in (["base", "mit"] if ed else ["base"]):
            if ed:
                ed.set(v)
            h = C.hidden_last(model, ids)[len(p_ids) - 1:]
            m, lp, act = C.close_margin(model, h, close_id, targets=tg)
            print(f"{v}: mean NLL {float(-act.mean()):.3f} | first 20 NLL {float(-act[:20].mean()):.3f} | mean close margin {float(m.mean()):.2f} | max {float(m.max()):.2f}")
        if ed:
            ed.set("base")
        bad = C.enc(tok, C.prompt_text(tok, probs[r['id']]['problem'], thinking=False))
        ids2 = torch.tensor([bad + x_ids], device=model.device)
        h = C.hidden_last(model, ids2)[len(bad) - 1:]
        _, _, act = C.close_margin(model, h, close_id, targets=tg)
        print(f"base, thinking-off prompt (틀린 프롬프트 대조): first 20 NLL {float(-act[:20].mean()):.3f}")
    except Exception:
        traceback.print_exc()

    try:
        step("8. short greedy generation, gsm8k_test_0")
        x = torch.tensor([C.enc(tok, pt)], device=model.device)
        for v in (["base", "mit"] if ed else ["base"]):
            if ed:
                ed.set(v)
            out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=120, do_sample=False,
                                 pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
            print(f"[{v}]", repr(tok.decode(out[0, x.shape[1]:])[:400]))
        if ed:
            ed.set("mit")
            y = torch.tensor([C.enc(tok, pt) + C.enc(tok, C.CUE)], device=model.device)
            out = model.generate(input_ids=y, attention_mask=torch.ones_like(y), max_new_tokens=24, do_sample=False,
                                 pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
            print("[forced answer with no thinking]", repr(tok.decode(out[0, y.shape[1]:])), "| gold", probs["gsm8k_test_0"]["answer"])
            ed.set("base")
    except Exception:
        traceback.print_exc()

    try:
        step("9. timing")
        for n in (2000, 8000, 16000):
            z = torch.randint(1000, 5000, (1, n), device=model.device)
            torch.cuda.synchronize()
            t0 = time.time()
            C.hidden_last(model, z)
            torch.cuda.synchronize()
            print(n, "tokens:", round(time.time() - t0, 2), "s | peak mem", round(torch.cuda.max_memory_allocated() / 2**30, 1), "GiB")
    except Exception:
        traceback.print_exc()
    print("\nSMOKE DONE", flush=True)


if __name__ == "__main__":
    main()
