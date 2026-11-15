"""공통 유틸: 경로, 모델 로드, MIT 편집 적용, 공개 풀이 기록 로드, 프롬프트 재구성, 답 추출.

용어
- 원 모델(base): Qwen/Qwen3.5-0.8B 그대로.
- MIT 편집(mit): MIT 공개 편집(runs/p50c/edit). 15층·19층 MLP down_proj의 64개 열씩을 바꾼다.
- aX (예: a1, a0.5): MIT 편집이 바꾼 양(delta)의 X/6배만 적용한 가중치. MIT 편집이 루프 방향을
  6배 뺀 것이라면 a1은 그 방향을 1배만 뺀 것(지우기)에 해당한다. 6배라는 값은 PoC 문서의 서술이며,
  smoke 단계에서 delta의 rank와 배수를 직접 확인한다.
"""
import glob
import gzip
import hashlib
import json
import os
import re

import numpy as np
import torch

MODEL_ID = os.environ.get("CWOT_MODEL_ID", "Qwen/Qwen3.5-0.8B")
MODEL_REV = os.environ.get("CWOT_MODEL_REV", "2fc06364715b967f1860aea9cf38778875588b17")
ROOT = os.environ.get("CWOT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MIT = os.environ.get("CWOT_MIT", os.path.join(ROOT, "data", "mit", "exp8_math"))
BENCH = os.path.join(ROOT, "data", "bench")
OUT = os.path.join(ROOT, "out")
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
THINK_CLOSE = "</think>"
CUE = "</think>\n\nThe final answer is \\boxed{"


# ---------------------------------------------------------------- 모델
def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)


def load_model(dtype=torch.bfloat16, device="cuda"):
    import transformers
    last = None
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        for kw in ({"dtype": dtype}, {"torch_dtype": dtype}):
            try:
                model = cls.from_pretrained(MODEL_ID, revision=MODEL_REV, **kw)
                model.to(device).eval()
                for p in model.parameters():
                    p.requires_grad_(False)
                return model
            except TypeError as e:      # dtype 인자 이름이 버전마다 다르다
                last = e
            except Exception as e:      # 이 클래스로는 안 열리면 다음 클래스로
                last = e
                break
    raise RuntimeError(f"모델을 열지 못했다: {last!r}")


def find_param(model, manifest_name):
    """manifest의 텐서 이름(model.language_model.layers.15.mlp.down_proj.weight)에 해당하는 파라미터.
    로드한 클래스에 따라 접두어가 다를 수 있어서 'layers.' 뒤쪽으로 맞춘다. 하나만 맞아야 한다."""
    suffix = manifest_name[manifest_name.index("layers."):]
    hits = [(n, p) for n, p in model.named_parameters() if n.endswith(suffix)]
    if len(hits) != 1:
        raise RuntimeError(f"{manifest_name}: {len(hits)}개가 맞음 {[n for n, _ in hits][:5]}")
    return hits[0]


def bf16_bits_of(t):
    """torch 텐서 -> bfloat16 비트(uint16) numpy 배열."""
    return t.detach().to("cpu", torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint16).copy()


def f32_of_bits(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def sha(bits):
    return hashlib.sha256(np.ascontiguousarray(bits).tobytes()).hexdigest()


class Editor:
    """원 가중치를 기억해 두고, 이름으로 가중치 변형을 바꿔 끼운다. 한 번에 한 변형만 적용된다."""

    def __init__(self, model, edit_dir=None, strict=True):
        self.model = model
        edit_dir = edit_dir or os.path.join(MIT, "runs", "p50c", "edit")
        man = json.load(open(os.path.join(edit_dir, "manifest.json")))
        z = np.load(os.path.join(edit_dir, "edit.npz"))
        self.items = []
        for i, t in enumerate(man["tensors"]):
            name, p = find_param(model, t["name"])
            base_bits = bf16_bits_of(p)
            ok_base = sha(base_bits) == t["base_sha256"]
            ed_bits = base_bits.copy().reshape(-1)
            ed_bits[z[f"t{i}_idx"]] = z[f"t{i}_val"]
            ed_bits = ed_bits.reshape(base_bits.shape)
            ok_ed = sha(ed_bits) == t["edited_sha256"]
            if strict and not (ok_base and ok_ed):
                raise RuntimeError(f"{name}: 해시 불일치 base={ok_base} edited={ok_ed}")
            base = torch.from_numpy(f32_of_bits(base_bits).copy())
            delta = torch.from_numpy(f32_of_bits(ed_bits).copy()) - base
            cols = torch.nonzero(delta.abs().sum(0) > 0).flatten()
            self.items.append({"name": name, "param": p, "base": base, "delta": delta, "cols": cols,
                               "hash_base": ok_base, "hash_edited": ok_ed, "manifest": t})
        self.current = "base"

    def describe(self):
        """delta의 구조를 요약한다: 바뀐 열 수, rank-1에 얼마나 가까운지, 루프 방향 u, 추정 배수."""
        out = []
        for it in self.items:
            d = it["delta"][:, it["cols"]].double()
            w = it["base"][:, it["cols"]].double()
            s = torch.linalg.svdvals(d)
            u = torch.linalg.svd(d, full_matrices=False)[0][:, 0]
            proj = u @ w                      # 각 열이 u 방향으로 쓰던 양
            dproj = u @ d                     # 편집이 u 방향으로 바꾼 양
            alpha = float(-(dproj @ proj) / (proj @ proj))
            out.append({"name": it["name"], "n_cols": int(it["cols"].numel()),
                        "rank1_energy": float(s[0] ** 2 / (s ** 2).sum()),
                        "alpha_est": alpha, "delta_rel": float(d.norm() / it["base"].double().norm()),
                        "hash_base": it["hash_base"], "hash_edited": it["hash_edited"]})
        return out

    def directions(self):
        """텐서별 루프 방향 u (delta의 첫 왼쪽 특이벡터, hidden 차원)."""
        return {it["name"]: torch.linalg.svd(it["delta"][:, it["cols"]].double(), full_matrices=False)[0][:, 0].float()
                for it in self.items}

    def set(self, variant):
        """variant: 'base' | 'mit' | 'a<배수>' (예: a1, a0.5, a3)."""
        if variant == "base":
            k = 0.0
        elif variant == "mit":
            k = 1.0
        elif re.fullmatch(r"a[0-9.]+", variant):
            k = float(variant[1:]) / 6.0
        else:
            raise ValueError(variant)
        for it in self.items:
            p = it["param"]
            new = it["base"] + k * it["delta"]
            p.data.copy_(new.to(p.device, p.dtype))
        self.current = variant


# ---------------------------------------------------------------- 공개 풀이 기록
def load_records(run, ds):
    """run: 'unedited' | 'p50c'. ds: 'gsm8k' | 'math500'. 문제마다 첫 응답 하나(200개)."""
    rows = []
    for d in sorted(glob.glob(os.path.join(MIT, "eval", run, "parts", f"{ds}_s*"))):
        per = {r["id"]: r for r in json.load(open(os.path.join(d, f"{ds}_per_problem.json")))}
        with gzip.open(os.path.join(d, f"{ds}_first_samples.jsonl.gz"), "rt") as f:
            for line in f:
                r = json.loads(line)
                r["correct_first"] = int(per[r["id"]]["correct"][0])
                r["pred_first"] = per[r["id"]].get("pred_first")
                r["run"], r["ds"] = run, ds
                rows.append(r)
    return rows


def load_problems(ds):
    """id -> {'problem', 'answer', (math500: 'level', 'subject')}. 처음 한 번 받아서 data/bench에 저장한다."""
    path = os.path.join(BENCH, f"{ds}.jsonl")
    if not os.path.exists(path):
        from datasets import load_dataset
        os.makedirs(BENCH, exist_ok=True)
        if ds == "gsm8k":
            rows = [{"id": f"gsm8k_test_{i}", "problem": r["question"],
                     "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
                    for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="test"))]
        elif ds == "math500":
            rows = [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"],
                     "level": r.get("level"), "subject": r.get("subject")}
                    for r in load_dataset("HuggingFaceH4/MATH-500", split="test")]
        else:
            raise ValueError(ds)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {r["id"]: r for r in map(json.loads, open(path))}


def prompt_text(tok, problem, thinking=True):
    return tok.apply_chat_template([{"role": "user", "content": problem + "\n" + INSTRUCTION}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def enc(tok, text):
    return tok(text, add_special_tokens=False).input_ids


def split_thinking(text):
    """(생각 글, 답 글, 생각을 닫았는가)."""
    i = text.find(THINK_CLOSE)
    if i < 0:
        return text, "", False
    return text[:i], text[i + len(THINK_CLOSE):], True


_NL_CACHE = {}


def newline_ids(tok):
    """디코드하면 줄바꿈이 들어 있는 토큰 id 집합. 문장·줄이 끝나는 자리를 찾는 데 쓴다."""
    key = id(tok)
    if key not in _NL_CACHE:
        path = os.path.join(ROOT, "data", "newline_ids.json")
        if os.path.exists(path):
            _NL_CACHE[key] = set(json.load(open(path)))
        else:
            n = len(tok)
            ids = [i for i in range(n) if "\n" in tok.decode([i])]
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump(ids, open(path, "w"))
            _NL_CACHE[key] = set(ids)
    return _NL_CACHE[key]


def line_ends(tok, ids):
    """ids 안에서 줄이 끝나는 자리. 값 p는 '앞에서 p개 토큰까지'가 줄 끝이라는 뜻이다(1..len)."""
    nl = newline_ids(tok)
    return [i + 1 for i, t in enumerate(ids) if t in nl]


# ---------------------------------------------------------------- 답 추출과 비교
def _box_at(text, idx):
    j = text.find("{", idx)
    if j < 0 or text[idx:j].strip() not in ("\\boxed", "\\fbox"):
        return None
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                s = text[j + 1:k].strip()
                return s or None
        k += 1
    return None


def last_boxed(text):
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    return _box_at(text, idx)


def close_brace(text):
    """'\\boxed{' 바로 뒤에 이어 쓴 글에서, 그 상자가 닫히기 전까지의 내용. 닫히지 않으면 None."""
    depth = 1
    for k, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[:k].strip() or None
    return None


def norm_answer(s):
    if s is None:
        return None
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    for a in ("\\left", "\\right", "\\!", "\\,", "\\ ", "\\$", "$", "^\\circ", "^{\\circ}", "\\%", "%"):
        s = s.replace(a, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace(" ", "").rstrip(".")
    t = s.replace(",", "")
    try:
        v = float(t)
        return repr(round(v, 6) + 0.0)
    except ValueError:
        return s


_MV = None


def same_answer(pred, gold):
    """정규화한 문자열 비교. math-verify가 설치되어 있으면 그것도 쓴다."""
    global _MV
    if pred is None or gold is None:
        return False
    if norm_answer(pred) == norm_answer(gold):
        return True
    if _MV is None:
        try:
            import math_verify
            _MV = math_verify
        except Exception:
            _MV = False
    if _MV:
        try:
            return bool(_MV.verify(_MV.parse("\\boxed{" + gold + "}"), _MV.parse("\\boxed{" + pred + "}")))
        except Exception:
            return False
    return False


# ---------------------------------------------------------------- 순전파 조각
@torch.no_grad()
def hidden_last(model, ids):
    """마지막 정규화까지 거친 hidden state [T, d].
    로짓 전체(T x 어휘 24.8만)를 한 번에 만들지 않으려고 본체만 부른다.
    CWOT_HIDDEN=full 이면 모델 전체를 불러 hidden_states[-1]을 쓴다(smoke에서 둘이 다르다고 나올 때)."""
    if os.environ.get("CWOT_HIDDEN", "base") == "full":
        out = model(input_ids=ids, output_hidden_states=True)
        return out.hidden_states[-1][0]
    out = model.base_model(input_ids=ids)
    return out.last_hidden_state[0]


@torch.no_grad()
def close_margin(model, h, close_id, targets=None, chunk=1024):
    """자리마다 (닫기 여유, </think>의 로그확률, 실제 다음 토큰의 로그확률 또는 None).
    닫기 여유 = </think> 점수 - 나머지 토큰 중 최고 점수. 0보다 크면 그 자리에서 닫는 것을 가장 선호한다.
    targets: h의 각 자리 다음에 실제로 온 토큰 id (길이는 h보다 짧아도 된다)."""
    head = model.get_output_embeddings()
    margins, logps, acts = [], [], []
    for s in range(0, h.shape[0], chunk):
        lg = head(h[s:s + chunk]).float()
        lc = lg[:, close_id]
        top = lg.topk(2, dim=-1)
        other = torch.where(top.indices[:, 0] == close_id, top.values[:, 1], top.values[:, 0])
        lse = torch.logsumexp(lg, dim=-1)
        margins.append(lc - other)
        logps.append(lc - lse)
        if targets is not None:
            t = targets[s:s + chunk]
            if t.numel():
                acts.append(lg[:t.numel()].gather(1, t[:, None])[:, 0] - lse[:t.numel()])
    return torch.cat(margins), torch.cat(logps), (torch.cat(acts) if acts else None)
