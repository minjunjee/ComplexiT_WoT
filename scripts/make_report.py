"""실험 1·4의 표와 그림을 만든다(GPU, torch 불필요).

  python scripts/make_report.py <입력 폴더> <출력 폴더>
입력 폴더: exp1_<run>_<ds>[_mismatch].jsonl (실험 1의 기록별 결과), exp4_summary.json (cwot.ana 가 만든 실험 4 요약)
출력: fig1_forced_accuracy.png, fig2_settled_point.png, fig3_close_margin.png, tables.md, numbers.json
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

for cand in ("Noto Sans CJK KR", "AppleGothic", "NanumGothic", "Noto Sans CJK JP"):
    if any(f.name == cand for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
RNG = np.random.default_rng(0)
NAMES = {"p50c": "편집한 모델", "unedited": "원래 모델"}
COL = {"p50c": "#1f6fb4", "unedited": "#c2410c", "mismatch": "#6b7280"}


def ci(x, f=np.median, n=4000):
    x = np.asarray(x, float)
    s = [f(RNG.choice(x, len(x))) for _ in range(n)]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def at(r, t):
    """생각을 t 토큰까지 읽혔을 때의 강제 답: t 이하에서 가장 늦은 자리의 답."""
    c = [x for x in r["cuts"] if x["cut"] <= t]
    return c[-1] if c else r["cuts"][0]


def stats(rows):
    n = len(rows)
    used = np.array([r["n_think_used"] for r in rows], float)
    st = [r for r in rows if r["settled_cut"] is not None and r["n_think_used"]]
    frac = np.array([r["settled_frac"] for r in st])
    # 답이 정해진 지점은 "처음으로 끝의 답과 같아지고 그 뒤로 바뀌지 않는 자리"다. 자리를 띄엄띄엄 봤으므로
    # 실제 지점은 그 자리와 바로 앞 자리 사이에 있다. 앞 자리를 아래쪽 경계로 같이 적는다.
    lo = []
    for r in st:
        cs = [x["cut"] for x in r["cuts"]]
        j = cs.index(r["settled_cut"])
        lo.append((cs[j - 1] if j > 0 else 0) / r["n_think_used"])
    half_fix = sum((not at(r, 0.5 * r["n_think_used"])["ok_gold"]) and r["cuts"][-1]["ok_gold"] for r in rows)
    half_break = sum(at(r, 0.5 * r["n_think_used"])["ok_gold"] and (not r["cuts"][-1]["ok_gold"]) for r in rows)
    prog = np.linspace(0, 1, 21)
    acc_prog = [float(np.mean([at(r, p * r["n_think_used"])["ok_gold"] for r in rows])) for p in prog]
    toks = [0, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
    acc_tok = [float(np.mean([at(r, t)["ok_gold"] for r in rows])) for t in toks]
    closed = [r for r in rows if r["closed"]]
    own = [r for r in closed if r["final"] is not None and r["n_think"] <= r["n_think_used"]]
    a, b = ci(frac)
    return {
        "n": n, "n_closed": len(closed),
        "think_tokens_median": float(np.median([r["n_think"] for r in rows])),
        "settled_tokens_median": float(np.median([r["settled_cut"] for r in st])),
        "settled_frac_median": float(np.median(frac)), "settled_frac_ci95": [a, b],
        "settled_frac_lower_bound_median": float(np.median(lo)),
        "after_settled_share_median": float(1 - np.median(frac)),
        "after_settled_share_token_weighted": float(sum(r["n_think_used"] - r["settled_cut"] for r in st) / sum(r["n_think_used"] for r in st)),
        "acc_no_thinking": float(np.mean([r["cuts"][0]["ok_gold"] for r in rows])),
        "acc_forced_at_end": float(np.mean([r["cuts"][-1]["ok_gold"] for r in rows])),
        "acc_forced_at_end_ci95": list(ci([r["cuts"][-1]["ok_gold"] for r in rows], np.mean)),
        "acc_public_first_sample": float(np.mean([r["correct_first"] for r in rows])),
        # 모델이 스스로 닫고 \\boxed 답을 냈고, 생각이 상한(16,000 토큰)에서 잘리지 않은 기록만 비교한다.
        "forced_end_equals_own_final": float(np.mean([r["cuts"][-1]["eq_final"] for r in own])) if own else None, "n_own_final": len(own),
        "wrong_at_half_right_at_end": int(half_fix), "right_at_half_wrong_at_end": int(half_break),
        "progress": [float(p) for p in prog], "acc_by_progress": acc_prog, "tokens": toks, "acc_by_tokens": acc_tok,
        "settled_fracs": [float(x) for x in frac],
    }


def main(src, dst):
    os.makedirs(dst, exist_ok=True)
    e1 = {}
    for p in sorted(glob.glob(os.path.join(src, "exp1_*.jsonl"))):
        rows = [json.loads(l) for l in open(p) if l.strip()]
        if rows:
            e1[os.path.basename(p)[5:-6]] = stats(rows)
    e4 = json.load(open(os.path.join(src, "exp4_summary.json"))) if os.path.exists(os.path.join(src, "exp4_summary.json")) else {}
    json.dump({"exp1": e1, "exp4": e4}, open(os.path.join(dst, "numbers.json"), "w"), indent=1, ensure_ascii=False)

    for ds, dsname in (("gsm8k", "GSM8K"), ("math500", "MATH-500")):
        keys = [k for k in (f"p50c_{ds}", f"unedited_{ds}") if k in e1]
        if not keys:
            continue
        # 그림 1: 생각을 어디까지 읽혔을 때 답을 맞히는가
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for k in keys:
            run = k.split("_")[0]; s = e1[k]
            ax[0].plot(np.array(s["progress"]) * 100, np.array(s["acc_by_progress"]) * 100, "-o", ms=3.5, color=COL[run], label=f"{NAMES[run]} (n={s['n']})")
            ax[0].axhline(s["acc_public_first_sample"] * 100, color=COL[run], ls=":", lw=1.2)
            ax[1].plot(range(len(s["tokens"])), np.array(s["acc_by_tokens"]) * 100, "-o", ms=3.5, color=COL[run], label=NAMES[run])
        mk = f"p50c_{ds}_mismatch"
        if mk in e1:
            s = e1[mk]
            ax[0].plot(np.array(s["progress"]) * 100, np.array(s["acc_by_progress"]) * 100, "--", color=COL["mismatch"], label=f"대조: 다른 문제의 생각을 넣음 (n={s['n']})")
        ax[0].plot([], [], ":", color="#374151", label="점선: 모델이 스스로 낸 첫 응답의 정답률 (같은 색)")
        ax[0].set_xlabel("읽힌 생각의 양 (그 기록의 생각 길이 대비 %)"); ax[0].set_ylabel("끊고 답을 시켰을 때 정답률 (%)")
        ax[0].set_title(f"{dsname}: 생각을 중간에서 끊고 답을 시키면"); ax[0].legend(fontsize=8, loc="upper left"); ax[0].set_ylim(0, 100); ax[0].set_xlim(0, 100)
        ax[1].set_xticks(range(len(e1[keys[0]]["tokens"]))); ax[1].set_xticklabels([str(t) for t in e1[keys[0]]["tokens"]], fontsize=8)
        ax[1].set_xlabel("읽힌 생각의 양 (토큰)"); ax[1].set_title("같은 결과를 토큰 수로 본 것"); ax[1].set_ylim(0, 100); ax[1].legend(fontsize=8.5, loc="lower right")
        for a in ax:
            a.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(os.path.join(dst, f"fig1_forced_accuracy_{ds}.png"), dpi=170); plt.close(fig)

        # 그림 2: 답이 정해진 지점의 분포
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        for k in keys:
            run = k.split("_")[0]; s = e1[k]; x = np.sort(s["settled_fracs"]) * 100
            ax.step(x, np.arange(1, len(x) + 1) / len(x) * 100, where="post", color=COL[run], label=f"{NAMES[run]} (중앙값 {s['settled_frac_median'] * 100:.0f}%)")
        ax.set_xlabel("답이 정해진 지점 (생각 길이 대비 %)"); ax.set_ylabel("기록의 누적 비율 (%)")
        ax.set_title(f"{dsname}: 답은 생각의 어디쯤에서 정해지는가"); ax.grid(alpha=0.25); ax.legend(fontsize=8.5, loc="lower right"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        fig.tight_layout(); fig.savefig(os.path.join(dst, f"fig2_settled_point_{ds}.png"), dpi=170); plt.close(fig)

        # 그림 3: 닫기 여유
        k = f"p50c_{ds}"
        if k in e4:
            v = e4[k]; var = v["variants"]; mult = [0 if q == "base" else 6 if q == "mit" else float(q[1:]) for q in var]
            cm = plt.cm.viridis(np.linspace(0.0, 0.9, len(var)))
            fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
            dec = np.arange(5, 100, 10)
            for i, q in enumerate(var):
                ax[0].plot(dec, v["margin_by_progress_decile"][i], "-o", ms=3, color=cm[i], label=f"{mult[i]:g}배" if mult[i] else "원래 가중치")
            ax[0].set_xlabel("생각이 진행된 정도 (%)"); ax[0].set_ylabel("닫기 여유 (줄 끝 자리 평균)")
            ax[0].set_title("생각 도중: 닫기는 줄곧 크게 밀린다")
            lo, hi = ax[0].get_ylim(); ax[0].set_ylim(lo - 0.28 * (hi - lo), hi)
            ax[0].legend(fontsize=8, title="편집 배수", title_fontsize=8, ncol=3, loc="lower center")
            base = np.array(v["margin_by_progress_decile"][0])
            for i, q in enumerate(var[1:], 1):
                ax[1].plot(dec, np.array(v["margin_by_progress_decile"][i]) - base, "-o", ms=3, color=cm[i], label=f"{mult[i]:g}배")
            ax[1].set_xlabel("생각이 진행된 정도 (%)"); ax[1].set_ylabel("닫기 여유가 오른 폭 (원래 가중치 대비)")
            ax[1].set_title("편집이 올려 주는 폭은 뒤로 갈수록 커진다"); ax[1].legend(fontsize=8, title="편집 배수", title_fontsize=8)
            ax[2].plot(mult, v["margin_at_actual_close"], "-o", color="#111827", label=f"실제로 닫은 자리 (n={v['n_closed']})")
            ax[2].plot(mult, v["margin_mean_line_ends"], "-o", color="#9ca3af", label="생각 도중의 줄 끝 자리 평균")
            ax[2].axhline(0, color="k", lw=0.8, ls=":"); ax[2].text(3.0, -4.2, "점선(0)보다 위: 닫기가 1순위", fontsize=8)
            ax[2].set_xlabel("편집 배수 (0 = 원래 가중치, 6 = 공개된 편집)"); ax[2].set_ylabel("닫기 여유")
            ax[2].set_title("닫은 자리는 원래 모델에서도 문턱 바로 아래"); ax[2].legend(fontsize=8)
            for a in ax:
                a.grid(alpha=0.25)
            fig.suptitle(f"{dsname}, 편집한 모델이 쓴 글 {v['n_records']}건을 가중치만 바꿔 다시 읽힌 결과", fontsize=10.5)
            fig.tight_layout(); fig.savefig(os.path.join(dst, f"fig3_close_margin_{ds}.png"), dpi=170); plt.close(fig)

    # 그림 5: 6배 편집이 닫기 여유를 올려 주는 폭. 편집한 모델의 글과 원래 모델의 글을 나란히.
    for ds, dsname in (("gsm8k", "GSM8K"), ("math500", "MATH-500")):
        ks = [k for k in (f"p50c_{ds}", f"unedited_{ds}") if k in e4]
        if len(ks) < 2:
            continue
        fig, ax = plt.subplots(figsize=(5.8, 4.0))
        dec = np.arange(5, 100, 10)
        for k in ks:
            run = k.split("_")[0]; b = e4[k]["margin_by_progress_decile"]
            lab = f"{NAMES[run]}이 쓴 글 ({e4[k]['n_records']}건)" + (", 앞 16,000토큰" if run == "unedited" else "")
            ax.plot(dec, np.array(b[-1]) - np.array(b[0]), "-o", ms=4, color=COL[run], label=lab)
        ax.set_xlabel("생각이 진행된 정도 (%)"); ax.set_ylabel("6배 편집이 닫기 여유를 올린 폭")
        ax.set_title(f"{dsname}: 편집이 미는 힘은 생각이 길어질수록 커진다", fontsize=10.5); ax.set_ylim(0, None)
        ax.grid(alpha=0.25); ax.legend(fontsize=8.5, loc="lower right")
        fig.tight_layout(); fig.savefig(os.path.join(dst, f"fig5_edit_push_{ds}.png"), dpi=170); plt.close(fig)

    # 그림 4: 기록 하나의 닫기 여유를 처음부터 끝까지 그린 예
    for ds in ("gsm8k", "math500"):
        pe = os.path.join(src, f"exp4_examples_p50c_{ds}.json")
        if not os.path.exists(pe):
            continue
        ex = [e for e in json.load(open(pe)) if e["close_pos"] > 3000][:3]
        if not ex:
            continue
        fig, ax = plt.subplots(1, len(ex), figsize=(4.9 * len(ex), 3.9), squeeze=False)
        for a, e in zip(ax[0], ex):
            x = np.array(e["ends"]); cp = e["close_pos"]
            body = x < cp
            a.plot(x[body], np.array(e["margin"][0])[body], lw=0.7, color="#9ca3af", label="원래 가중치")
            a.plot(x[body], np.array(e["margin"][-1])[body], lw=0.7, color="#1f6fb4", label="공개된 편집(6배)")
            i = int(np.argmax(x == cp)) if (x == cp).any() else len(x) - 1
            a.plot([cp], [e["margin"][0][i]], "o", color="#4b5563", ms=5); a.plot([cp], [e["margin"][-1][i]], "o", color="#1f6fb4", ms=5)
            a.axvline(e["settled_cut"], color="#c2410c", ls="--", lw=1); a.axhline(0, color="k", ls=":", lw=0.8)
            a.text(e["settled_cut"], a.get_ylim()[0] + 1, " 답이 정해진 지점", color="#c2410c", fontsize=8, va="bottom")
            a.annotate("실제로 닫은 자리", (cp, e["margin"][-1][i]), xytext=(-78, -16), textcoords="offset points", fontsize=8)
            a.set_title(f"{e['id']} (생각 {cp:,} 토큰)", fontsize=9.5); a.set_xlabel("생각 안에서의 위치 (토큰)"); a.grid(alpha=0.25)
        ax[0][0].set_ylabel("닫기 여유 (줄 끝 자리)"); ax[0][0].legend(fontsize=8, loc="lower right")
        fig.tight_layout(); fig.savefig(os.path.join(dst, f"fig4_margin_examples_{ds}.png"), dpi=170); plt.close(fig)

    # 표
    L = ["# 실험 1·4 숫자 요약 (자동 생성)\n", "## 실험 1: 끊어서 답 시켜 보기\n",
         "| 항목 | " + " | ".join(e1.keys()) + " |", "|---|" + "---|" * len(e1)]
    pc = lambda x: "-" if x is None else f"{100 * x:.1f}%"
    rowsdef = [("기록 수", lambda s: str(s["n"])), ("스스로 생각을 닫은 기록", lambda s: str(s["n_closed"])),
               ("생각 길이 중앙값(토큰)", lambda s: f"{s['think_tokens_median']:.0f}"),
               ("답이 정해진 지점 중앙값(토큰)", lambda s: f"{s['settled_tokens_median']:.0f}"),
               ("답이 정해진 지점 중앙값(생각 길이 대비)", lambda s: f"{pc(s['settled_frac_median'])} [{pc(s['settled_frac_ci95'][0])}, {pc(s['settled_frac_ci95'][1])}]"),
               ("같은 값의 아래쪽 경계(바로 앞 자리 기준)", lambda s: pc(s["settled_frac_lower_bound_median"])),
               ("답이 정해진 뒤에 쓴 생각의 비율(중앙값)", lambda s: pc(s["after_settled_share_median"])),
               ("같은 비율, 토큰 수로 합산", lambda s: pc(s["after_settled_share_token_weighted"])),
               ("생각 없이 답을 시켰을 때 정답률", lambda s: pc(s["acc_no_thinking"])),
               ("생각 끝에서 답을 시켰을 때 정답률", lambda s: f"{pc(s['acc_forced_at_end'])} [{pc(s['acc_forced_at_end_ci95'][0])}, {pc(s['acc_forced_at_end_ci95'][1])}]"),
               ("모델이 스스로 낸 첫 응답의 정답률(공개 기록)", lambda s: pc(s["acc_public_first_sample"])),
               ("끝에서 시킨 답 = 모델이 스스로 낸 답 (비교 가능한 기록 수)", lambda s: f"{pc(s['forced_end_equals_own_final'])} ({s['n_own_final']})"),
               ("절반 지점에서 틀렸다가 끝에서 맞은 기록", lambda s: str(s["wrong_at_half_right_at_end"])),
               ("절반 지점에서 맞았다가 끝에서 틀린 기록", lambda s: str(s["right_at_half_wrong_at_end"]))]
    for name, f in rowsdef:
        L.append(f"| {name} | " + " | ".join(f(s) for s in e1.values()) + " |")
    L.append("\n## 실험 4: 닫기 여유\n")
    for k, v in e4.items():
        L += [f"### {k} (기록 {v['n_records']}건, 그중 스스로 닫은 기록 {v.get('n_closed')}건)\n",
              "| 항목 | " + " | ".join(v["variants"]) + " |", "|---|" + "---|" * len(v["variants"])]
        for name, key in [("생각 도중 줄 끝 자리의 닫기 여유 평균", "margin_mean_line_ends"), ("생각 도중 모든 자리의 평균", "margin_mean_all_positions"),
                          ("답이 정해지기 전 줄 끝 자리", "margin_mean_before_settled"), ("답이 정해진 뒤 줄 끝 자리", "margin_mean_after_settled"),
                          ("실제로 닫은 자리(평균)", "margin_at_actual_close"), ("실제로 닫은 자리(중앙값)", "margin_at_actual_close_median"),
                          ("닫은 자리에서 닫기가 1순위인 기록 비율", "share_close_preferred_at_actual_close"),
                          ("닫은 자리의 닫기 여유가 그 앞 모든 자리보다 높은 기록 비율", "close_pos_is_top_among_earlier"),
                          ("닫은 자리의 순위(앞 자리 중 자기보다 낮은 자리의 비율)", "close_pos_rank_among_earlier_positions"),
                          ("닫기 전에 닫기가 1순위인 자리가 하나라도 있는 기록 비율", "share_records_close_preferred_somewhere_before_close"),
                          ("원래 가중치 대비 변화: 닫은 자리", "change_vs_base_at_actual_close"), ("원래 가중치 대비 변화: 모든 자리", "change_vs_base_all_positions"),
                          ("글의 토큰당 NLL", "nll_mean")]:
            if v.get(key):
                L.append(f"| {name} | " + " | ".join("-" if x is None else f"{x:.3f}" for x in v[key]) + " |")
        L.append("")
    open(os.path.join(dst, "tables.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
