"""실험 1과 4의 결과를 요약한다(GPU 불필요). 결과가 일부만 있어도 있는 만큼 요약한다.

  python -m cwot.ana
출력: out/summary/exp1.json, out/summary/exp4.json 과 화면의 표
"""
import glob
import json
import os
import re

import numpy as np

from . import common as C

RNG = np.random.default_rng(0)


def boot_ci(x, f=np.median, n=2000):
    x = np.asarray(x, dtype=float)
    if len(x) < 5:
        return [None, None]
    s = [f(RNG.choice(x, len(x))) for _ in range(n)]
    return [round(float(np.percentile(s, 2.5)), 4), round(float(np.percentile(s, 97.5)), 4)]


def load_exp1():
    out = {}
    for p in sorted(glob.glob(os.path.join(C.OUT, "exp1", "*.jsonl"))):
        name = re.sub(r"_s\d+of\d+$", "", os.path.basename(p)[:-6])      # 나눠 돌린 조각은 합친다
        out.setdefault(name, []).extend(json.loads(l) for l in open(p))
    return out


def ana1(e1):
    res = {}
    for name, rows in e1.items():
        if not rows:
            continue
        sf = [r["settled_frac"] for r in rows if r["settled_frac"] is not None]
        closed = [r for r in rows if r["closed"]]
        sfc = [r["settled_frac"] for r in closed if r["settled_frac"] is not None]
        # 정답 기준: 강제 답이 정답이 되고 그 뒤로 계속 정답인 가장 이른 자리 (끝에서 정답인 기록만)
        gf = []
        for r in rows:
            ok = [c["ok_gold"] for c in r["cuts"]]
            if ok[-1] and r["n_think_used"]:
                j = len(ok) - 1
                while j > 0 and ok[j - 1]:
                    j -= 1
                gf.append(r["cuts"][j]["cut"] / r["n_think_used"])
        # 생각이 진행된 비율(10칸)별 강제 답 정답률
        bins = [[] for _ in range(10)]
        for r in rows:
            for c in r["cuts"]:
                if r["n_think_used"]:
                    bins[min(9, int(10 * c["cut"] / r["n_think_used"]))].append(c["ok_gold"])
        # 뒤쪽 생각이 답을 고친 경우: 생각 절반 지점 이전의 마지막 강제 답은 틀렸는데 끝에서는 맞은 기록
        fixed_late = 0
        for r in rows:
            half = [c for c in r["cuts"] if r["n_think_used"] and c["cut"] <= 0.5 * r["n_think_used"]]
            if half and (not half[-1]["ok_gold"]) and r["cuts"][-1]["ok_gold"]:
                fixed_late += 1
        res[name] = {
            "n": len(rows), "n_closed": len(closed), "n_stable": len(sf),
            "settled_frac_median": round(float(np.median(sf)), 4) if sf else None,
            "settled_frac_median_ci95": boot_ci(sf),
            "after_settled_share_median": round(1 - float(np.median(sf)), 4) if sf else None,
            "settled_frac_median_closed_only": round(float(np.median(sfc)), 4) if sfc else None,
            "settled_tokens_median": float(np.median([r["settled_cut"] for r in rows if r["settled_cut"] is not None])) if sf else None,
            "think_tokens_median": float(np.median([r["n_think"] for r in rows])),
            "gold_settled_frac_median": round(float(np.median(gf)), 4) if gf else None, "n_gold_settled": len(gf),
            "acc_no_thinking": round(float(np.mean([r["cuts"][0]["ok_gold"] for r in rows])), 4),
            "acc_forced_at_end": round(float(np.mean([r["cuts"][-1]["ok_gold"] for r in rows])), 4),
            "acc_public_first_sample": round(float(np.mean([r["correct_first"] for r in rows])), 4),
            "forced_end_equals_own_final": round(float(np.mean([r["cuts"][-1]["eq_final"] for r in closed])), 4) if closed else None,
            "fixed_in_second_half": fixed_late,
            "acc_by_progress_decile": [round(float(np.mean(b)), 3) if b else None for b in bins],
        }
    return res


def ana4(e1):
    res = {}
    for d in sorted(glob.glob(os.path.join(C.OUT, "exp4", "*"))):
        name = os.path.basename(d)
        settled = {r["id"]: r for r in e1.get(name, [])}
        files = sorted(glob.glob(os.path.join(d, "*.npz")))
        if not files:
            continue
        acc = None
        for f in files:
            z = np.load(f)
            meta = json.loads(str(z["meta"]))
            v = meta["variants"]
            m = z["margin"].astype(np.float64)                      # [V, n+1]
            n_think = meta["close_pos"] if meta["close_pos"] >= 0 else meta["n_used"]
            ends = z["line_ends"]
            # 실제로 닫은 자리는 따로 센다(margin_at_actual_close). 줄 끝 통계에서는 뺀다.
            ends = ends[ends < n_think] if meta["close_pos"] >= 0 else ends[ends <= n_think]
            if acc is None:
                acc = {"variants": v, "all": [], "pre": [], "post": [], "at_close": [], "pos_frac": [],
                       "pre_pos": [], "post_pos": [], "curve": [[[] for _ in range(20)] for _ in v], "nll": [],
                       "close_pct": [], "d_close": [], "d_body": [], "ex": [], "allpos": [], "mx": [], "anypos": [], "firstpos": [[] for _ in v], "prog": [[[] for _ in range(10)] for _ in v]}
            if len(ends) < 4:
                continue
            # 모든 자리(줄 끝이 아닌 자리 포함). 실제로 닫은 자리 자체는 뺀다.
            body = m[:, :max(n_think, 1)]
            acc["allpos"].append(body.mean(1)); acc["mx"].append(body.max(1))
            acc["anypos"].append((body > 0).any(1).astype(float))
            for q in range(len(v)):
                w = np.nonzero(body[q] > 0)[0]
                if len(w):
                    acc["firstpos"][q].append(w[0] / max(n_think, 1))
            for k, e in enumerate(ends):
                b = min(9, int(10 * e / max(n_think, 1)))
                for q in range(len(v)):
                    acc["prog"][q][b].append(m[q, e])
            mm = m[:, ends]
            acc["all"].append(mm.mean(1))
            acc["pos_frac"].append((mm > 0).mean(1))
            acc["nll"].append(z["nll"][:, :max(n_think, 1)].mean(1))
            acc["d_body"].append((body - body[0:1]).mean(1))          # 변형 0(원래 가중치) 대비, 생각 전체 자리의 평균 변화
            if meta["close_pos"] >= 0:
                cp = meta["close_pos"]
                acc["at_close"].append(m[:, cp])
                acc["d_close"].append(m[:, cp] - m[0, cp])             # 실제로 닫은 자리에서의 변화
                if cp > 0:   # 닫은 자리의 닫기 여유가 그 앞 모든 자리 가운데 몇 번째로 높은가(1.0 = 가장 높다)
                    acc["close_pct"].append((m[:, :cp] < m[:, cp:cp + 1]).mean(1))
            if len(acc["ex"]) < 6 and meta["close_pos"] >= 0 and meta["id"] in settled and settled[meta["id"]]["settled_cut"]:
                acc["ex"].append({"id": meta["id"], "close_pos": meta["close_pos"], "settled_cut": settled[meta["id"]]["settled_cut"],
                                  "ends": [int(e) for e in ends] + [int(meta["close_pos"])], "variants": v,     # 그림용: 닫은 자리를 끝에 붙인다
                                  "margin": [[round(float(x), 2) for x in list(m[q, ends]) + [m[q, meta["close_pos"]]]] for q in range(len(v))]})
            s = settled.get(meta["id"])
            if s and s["settled_cut"] is not None and 0 < s["settled_cut"] < n_think:
                sc = s["settled_cut"]
                pre, post = mm[:, ends < sc], mm[:, ends >= sc]
                if pre.shape[1] and post.shape[1]:
                    acc["pre"].append(pre.mean(1)); acc["post"].append(post.mean(1))
                    acc["pre_pos"].append((pre > 0).mean(1)); acc["post_pos"].append((post > 0).mean(1))
                    # 답이 정해진 지점을 0으로: 앞은 -1..0 (지점까지의 비율), 뒤는 0..1 (지점 뒤 남은 길이의 비율)
                    for k, e in enumerate(ends):
                        x = (e - sc) / sc if e < sc else (e - sc) / max(n_think - sc, 1)
                        b = min(19, max(0, int((x + 1) * 10)))
                        for q in range(len(v)):
                            acc["curve"][q][b].append(mm[q, k])
        if acc is None or not acc["all"]:
            continue
        mean = lambda a: [round(float(x), 3) for x in np.mean(np.stack(a), 0)] if a else None
        json.dump(acc["ex"], open(os.path.join(C.OUT, "summary", f"exp4_examples_{name}.json"), "w"))
        res[name] = {
            "variants": acc["variants"], "n_records": len(acc["all"]), "n_with_settled": len(acc["pre"]),
            "margin_mean_line_ends": mean(acc["all"]), "share_positive_line_ends": mean(acc["pos_frac"]),
            "margin_mean_all_positions": mean(acc["allpos"]), "margin_max_before_close": mean(acc["mx"]),
            "share_records_close_preferred_somewhere_before_close": mean(acc["anypos"]),
            "first_close_preferred_frac_median": [round(float(np.median(x)), 4) if x else None for x in acc["firstpos"]],
            "margin_by_progress_decile": [[round(float(np.mean(b)), 3) if b else None for b in row] for row in acc["prog"]],
            "margin_mean_before_settled": mean(acc["pre"]), "margin_mean_after_settled": mean(acc["post"]),
            "share_positive_before": mean(acc["pre_pos"]), "share_positive_after": mean(acc["post_pos"]),
            "margin_at_actual_close": mean(acc["at_close"]), "n_closed": len(acc["at_close"]),
            "margin_at_actual_close_median": [round(float(x), 3) for x in np.median(np.stack(acc["at_close"]), 0)] if acc["at_close"] else None,
            "share_close_preferred_at_actual_close": [round(float(x), 3) for x in (np.stack(acc["at_close"]) > 0).mean(0)] if acc["at_close"] else None,
            "close_pos_rank_among_earlier_positions": mean(acc["close_pct"]),
            "close_pos_is_top_among_earlier": [round(float(x), 3) for x in (np.stack(acc["close_pct"]) >= 1.0).mean(0)] if acc["close_pct"] else None,
            "change_vs_base_at_actual_close": mean(acc["d_close"]), "change_vs_base_all_positions": mean(acc["d_body"]),
            "nll_mean": mean(acc["nll"]),
            "curve_bins_-1_to_1": [[round(float(np.mean(b)), 3) if b else None for b in row] for row in acc["curve"]],
        }
    return res


def main():
    os.makedirs(os.path.join(C.OUT, "summary"), exist_ok=True)
    e1 = load_exp1()
    r1 = ana1(e1)
    json.dump(r1, open(os.path.join(C.OUT, "summary", "exp1.json"), "w"), indent=1, ensure_ascii=False)
    for k, v in r1.items():
        print(f"\n== exp1 {k}")
        for a, b in v.items():
            print(f"  {a}: {b}")
    r4 = ana4(e1)
    json.dump(r4, open(os.path.join(C.OUT, "summary", "exp4.json"), "w"), indent=1, ensure_ascii=False)
    for k, v in r4.items():
        print(f"\n== exp4 {k}  (variants {v['variants']})")
        for a, b in v.items():
            if a not in ("variants", "curve_bins_-1_to_1", "margin_by_progress_decile"):
                print(f"  {a}: {b}")
        print("  curve (답이 정해진 지점 = 가운데, 칸 20개):")
        for q, row in zip(v["variants"], v["curve_bins_-1_to_1"]):
            print("   ", q, row)
    print("\nANA DONE")


if __name__ == "__main__":
    main()
