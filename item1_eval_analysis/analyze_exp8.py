"""Item 1: is Exp 8's "math gain" mostly a loop fix?  Analysis of the released evaluation outputs only.

    python analyze_exp8.py <exp8_math/eval dir> <out dir>

Data used (all from lm_understand-logs/exp8_math/eval/<run>/parts/<set>_sKofN/):
  <set>_per_problem.json        per problem: correct = one 0/1 per sample
  <set>_first_samples.jsonl.gz  full text of the FIRST sample of every problem (thinking + answer)

Per-sample lengths / finish reasons are not released, so for first samples we use a proxy:
  closed = the text contains "</think>"  (a thinking that never closes ran into the 32,768-token cap).
We check this proxy against the released cut_share (scores.json) below.
"""
import collections
import glob
import gzip
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = ["unedited", "p10c", "p25c", "p50c"]
RUN_LABEL = {"unedited": "unedited", "p10c": "10 passes", "p25c": "25 passes", "p50c": "50 passes"}
MATH_SETS = ["gsm8k", "math500", "minerva", "olympiad", "amc23"]
ALL_SETS = MATH_SETS + ["aime24", "aime25", "mmlupro1k", "gpqa", "ifeval"]
NGRAM = 8


# ---------------------------------------------------------------- loading
def load_per_problem(ev, run, s):
    rows = []
    for f in sorted(glob.glob(f"{ev}/{run}/parts/{s}_s*/{s}_per_problem.json")):
        rows += json.load(open(f))
    return rows


def load_first(ev, run, s):
    rows = []
    for f in sorted(glob.glob(f"{ev}/{run}/parts/{s}_s*/{s}_first_samples.jsonl.gz")):
        rows += [json.loads(l) for l in gzip.open(f, "rt")]
    return rows


def released_cut_share(ev, run, s):
    cut = tot = 0
    for f in glob.glob(f"{ev}/{run}/parts/{s}_s*/scores.json"):
        d = json.load(open(f))[s]
        cut += d["cut"]
        tot += d["total"]
    return cut / tot if tot else np.nan


# ---------------------------------------------------------------- repetition measures
def loop_stats(text):
    """Word-level loop measures for one generation.

    dup_tail : share of the last 500 word-8-grams already seen earlier in the text (1.0 = pure loop)
    onset    : relative position (0..1) where the text turns into a loop: the earliest word i such that
               >= 90% of the 8-grams from i on are repeats.  NaN if it never does.
    period   : distance in words from the last 8-gram to its previous occurrence (loop length).
    ws_tail  : the last 2,000 characters are >= 90% whitespace (whitespace degeneration).
    """
    tail = text[-2000:]
    ws_tail = len(tail) > 0 and sum(c.isspace() for c in tail) / len(tail) >= 0.9
    w = text.split()
    n = len(w) - NGRAM + 1
    if n <= 0:
        return dict(words=len(w), dup_tail=0.0, onset=np.nan, period=np.nan, ws_tail=ws_tail)
    seen, last_pos = set(), {}
    dup = np.zeros(n, dtype=np.int8)
    period = np.nan
    for i in range(n):
        g = tuple(w[i:i + NGRAM])
        if g in seen:
            dup[i] = 1
            if i == n - 1:
                period = i - last_pos[g]
        seen.add(g)
        last_pos[g] = i
    dup_tail = float(dup[-500:].mean())
    suf = np.cumsum(dup[::-1])[::-1] / np.arange(n, 0, -1)       # mean of dup[i:]
    ok = np.where((suf >= 0.9) & (np.arange(n) < n - 200))[0]    # loop must last >= 200 words
    onset = ok[0] / n if len(ok) else np.nan
    return dict(words=len(w), dup_tail=dup_tail, onset=onset, period=period, ws_tail=ws_tail)


def classify(r):
    if r["closed"]:
        return "finished"
    if r["ws_tail"]:
        return "cut: whitespace"
    if r["dup_tail"] >= 0.8:
        return "cut: loop"
    return "cut: no loop"


# ---------------------------------------------------------------- main
def main(ev, out):
    os.makedirs(out, exist_ok=True)
    report = []
    say = lambda *a: (print(*a), report.append(" ".join(str(x) for x in a)))  # noqa: E731

    # A. benchmark accuracy over all samples (should match merged.json)
    acc = {}
    for run in RUNS:
        m = json.load(open(f"{ev}/{run}/merged.json"))
        for s in ALL_SETS:
            pp = load_per_problem(ev, run, s)
            a = np.mean([c for r in pp for c in r["correct"]]) if pp else np.nan
            acc[(run, s)] = dict(acc=a, merged_acc=m.get(s, {}).get("acc"), cut_share=m.get(s, {}).get("cut_share"))
    A = pd.DataFrame(acc).T
    A.index.names = ["run", "set"]
    A.to_csv(f"{out}/A_benchmark_acc.csv")
    say("## A. Accuracy (all samples) / cut share, from per_problem.json and merged.json")
    say(A["acc"].unstack(0)[RUNS].loc[ALL_SETS].round(3).to_string())
    say("max |recomputed - merged| =", float((A["acc"] - A["merged_acc"].astype(float)).abs().max()))
    say("\ncut share:")
    say(A["cut_share"].unstack(0)[RUNS].loc[ALL_SETS].astype(float).round(3).to_string())

    # B. first-sample table
    recs = []
    for run in RUNS:
        for s in MATH_SETS + ["mmlupro1k"]:
            pp = {r["id"]: r for r in load_per_problem(ev, run, s)}
            for r in load_first(ev, run, s):
                st = loop_stats(r["text"])
                recs.append(dict(run=run, set=s, id=r["id"], correct=pp[r["id"]]["correct"][0],
                                 closed="</think>" in r["text"], chars=len(r["text"]), **st))
    F = pd.DataFrame(recs)
    F["kind"] = F.apply(classify, axis=1)
    F.to_csv(f"{out}/B_first_samples.csv", index=False)

    say("\n## B0. Proxy check: share of first samples NOT closing </think>  vs released cut_share (all samples)")
    chk = F.groupby(["set", "run"])["closed"].apply(lambda x: 1 - x.mean()).unstack()[RUNS]
    rel = pd.DataFrame({run: {s: released_cut_share(ev, run, s) for s in chk.index} for run in RUNS})
    say(pd.concat({"proxy(first)": chk, "released": rel}, axis=1).round(3).to_string())

    say("\n## B1. What the first samples look like (share of each kind)")
    K = F.groupby(["run", "kind"]).size().unstack(fill_value=0)
    K = (K.T / K.sum(1)).T.loc[RUNS]
    say(K.round(3).to_string())

    say("\n## B2. Accuracy conditional on finishing (first samples, math sets)")
    M = F[F.set.isin(MATH_SETS)]
    cond = M.groupby(["run", "closed"])["correct"].agg(["mean", "size"]).unstack().loc[RUNS]
    say(cond.round(3).to_string())

    # C. paired transitions unedited -> each edit, keyed by (set,id)
    say("\n## C. Paired transitions of first samples, unedited -> edited (math sets)")
    base = M[M.run == "unedited"].set_index(["set", "id"])
    trans_rows = []
    for run in ["p10c", "p25c", "p50c"]:
        e = M[M.run == run].set_index(["set", "id"])
        j = base.join(e, lsuffix="_u", rsuffix="_e", how="inner")
        gained = (j.correct_e == 1) & (j.correct_u == 0)
        lost = (j.correct_e == 0) & (j.correct_u == 1)
        n = len(j)
        g_cut = (gained & ~j.closed_u).sum()
        both_closed = j[j.closed_u & j.closed_e]
        say(f"{run}: n={n}  gained={gained.sum()} (of which base was cut: {g_cut}, "
            f"{g_cut / max(gained.sum(), 1):.1%})  lost={lost.sum()}  net={gained.sum() - lost.sum()}")
        say(f"    problems where BOTH finished: n={len(both_closed)}  acc unedited={both_closed.correct_u.mean():.3f}"
            f"  edited={both_closed.correct_e.mean():.3f}")
        for (cu, ce), grp in j.groupby(["closed_u", "closed_e"]):
            trans_rows.append(dict(run=run, base_closed=cu, edit_closed=ce, n=len(grp),
                                   acc_u=grp.correct_u.mean(), acc_e=grp.correct_e.mean()))
        for s in MATH_SETS:
            js = j.loc[s]
            g = ((js.correct_e == 1) & (js.correct_u == 0))
            trans_rows.append(dict(run=run, set=s, n=len(js), gained=int(g.sum()),
                                   gained_from_cut=int((g & ~js.closed_u).sum()),
                                   lost=int(((js.correct_e == 0) & (js.correct_u == 1)).sum())))
    T = pd.DataFrame(trans_rows)
    T.to_csv(f"{out}/C_transitions.csv", index=False)
    say(T[T.set.isna()].drop(columns=["set", "gained", "gained_from_cut", "lost"]).round(3).to_string(index=False))
    say(T[T.set.notna()][["run", "set", "n", "gained", "gained_from_cut", "lost"]].to_string(index=False))

    # D. the loops themselves (unedited, cut, looping first samples)
    say("\n## D. Loops in the unedited model (cut first samples, math sets)")
    L = M[(M.run == "unedited") & (M.kind == "cut: loop")]
    say(f"n={len(L)}  onset (rel. position) median={L.onset.median():.3f}  IQR="
        f"[{L.onset.quantile(.25):.3f}, {L.onset.quantile(.75):.3f}]")
    say(f"loop period (words) median={L.period.median():.0f}  IQR=[{L.period.quantile(.25):.0f}, "
        f"{L.period.quantile(.75):.0f}]")
    say(f"words of correct finished samples, unedited median={M[(M.run=='unedited')&M.closed&(M.correct==1)].words.median():.0f}"
        f"  p50c median={M[(M.run=='p50c')&M.closed&(M.correct==1)].words.median():.0f}")

    # examples of loop tails, for reading
    ex = L.sample(min(5, len(L)), random_state=0)
    with open(f"{out}/D_loop_examples.md", "w") as fh:
        texts = {(r["id"]): r["text"] for s in MATH_SETS for r in load_first(ev, "unedited", s)}
        for _, r in ex.iterrows():
            t = texts[r["id"]]
            k = int(r.onset * len(t)) if r.onset == r.onset else len(t) - 1500
            fh.write(f"### {r.set} {r.id}  (onset {r.onset:.2f}, period {r.period} words)\n\n"
                     f"**just before the loop:**\n```\n{t[max(0, k - 1200):k]}\n```\n"
                     f"**the loop (end of text):**\n```\n{t[-1200:]}\n```\n\n")

    # ------------------------------------------------------------ figures
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    piv = A["acc"].unstack(0)[RUNS].loc[ALL_SETS].astype(float)
    piv.plot.bar(ax=ax[0], width=0.8)
    ax[0].set_ylabel("accuracy (all samples)")
    ax[0].set_title("Accuracy by benchmark")
    ax[0].legend([RUN_LABEL[r] for r in RUNS], fontsize=8)
    cs = A["cut_share"].astype(float)
    for run in RUNS:
        sub = A.loc[run].loc[ALL_SETS]
        ax[1].scatter(sub.cut_share.astype(float), sub.acc.astype(float), label=RUN_LABEL[run])
    ax[1].set_xlabel("share of answers cut at 32,768 tokens")
    ax[1].set_ylabel("accuracy")
    ax[1].set_title("Accuracy vs truncation (one dot = benchmark x run)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}/fig1_acc_and_cut.png", dpi=140)

    fig, ax = plt.subplots(figsize=(7, 4))
    K[["finished", "cut: loop", "cut: whitespace", "cut: no loop"]].rename(index=RUN_LABEL).plot.bar(
        stacked=True, ax=ax, color=["#4c9a5b", "#c0504d", "#e3a33b", "#8c8c8c"])
    ax.set_ylabel("share of first samples")
    ax.set_title("How generations end (math sets + MMLU-Pro)")
    fig.tight_layout()
    fig.savefig(f"{out}/fig2_endings.png", dpi=140)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].hist(L.onset.dropna(), bins=30, color="#c0504d")
    ax[0].set_xlabel("relative position where the loop starts")
    ax[0].set_title("Loop onset (unedited, cut samples)")
    ax[1].hist(np.log10(L.period.dropna().clip(lower=1)), bins=30, color="#555")
    ax[1].set_xlabel("log10 loop period (words)")
    ax[1].set_title("Loop length")
    fig.tight_layout()
    fig.savefig(f"{out}/fig3_loops.png", dpi=140)

    open(f"{out}/report.txt", "w").write("\n".join(report) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
