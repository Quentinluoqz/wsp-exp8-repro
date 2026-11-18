"""Item 4 driver for a Colab VM started with the colab CLI (same steps as item4_hand_direction.ipynb).
Run after item 3 on the same VM (it reuses its test problems and its base / p50c think35 generations):
    nohup python -u /content/run_item4.py > /content/wsp/item4.log 2>&1 &
Every step caches its output in /content/wsp/item4, so a rerun resumes.
"""
import gc
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/content")
import wsp_colab as W  # noqa: E402

pd.options.display.float_format = "{:.3f}".format
pd.options.display.width = 200
WORK = "/content/wsp"
OUT = f"{WORK}/item4"
os.makedirs(OUT, exist_ok=True)

N_PRACTICE_GSM, N_PRACTICE_MATH, N_SAMPLES = 150, 150, 2
LAYERS = list(range(8, 23))
EDIT_LAYERS = (15, 19)
K_TOKENS, MAX_CTX = 32, 8000
TOPK, ALPHAS = 64, (3.0, 6.0)
HELDOUT = 0.2
GEN_EDITS = ["hand_a6", "random_a6", "hand_a3"]


def say(*a):
    print(*a, flush=True)


exp8 = W.get_logs()
base = W.get_base()
practice_path = f"{OUT}/practice_problems.jsonl"
practice = W.load_practice(practice_path, exp8, N_PRACTICE_GSM, N_PRACTICE_MATH)
problems_path = f"{WORK}/problems_test.jsonl"
W.load_test(problems_path)

# ---- step 1: practice generations
say("== step 1: practice generations")
gen_path = f"{OUT}/practice_base.jsonl"
W.generate(base, practice_path, "think35_8k", gen_path, n=N_SAMPLES)
sc = W.score_file(gen_path, exp8)
say(sc.kind.value_counts().to_string())
say("loop onset (share of the answer): median", round(float(sc.onset_rel.median()), 3))

# ---- step 2: contexts
say("== step 2: contexts")
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(base)
ctx = W.tokenize_contexts(W.build_contexts(gen_path, sc), tok, practice, max_ctx_tokens=MAX_CTX)
rng = np.random.default_rng(0)
ids = sorted({c["id"] for c in ctx})
held = set(rng.choice(ids, int(len(ids) * HELDOUT), replace=False))
for c in ctx:
    c["split"] = "held" if c["id"] in held else "train"
kind = np.array([c["kind"] for c in ctx])
split = np.array([c["split"] for c in ctx])
say(pd.crosstab(pd.Series(kind, name="kind"), pd.Series(split, name="split")).to_string())
say("context length (tokens): median", int(np.median([len(c["ids"]) for c in ctx])))
json.dump([{k: c[k] for k in ("kind", "gid", "id", "split")} for c in ctx], open(f"{OUT}/contexts_index.json", "w"))

# ---- step 3: activations
say("== step 3: activations")
import torch  # noqa: E402

m, _ = W.load_hf(base)
act_path = f"{OUT}/acts.npz"
if os.path.exists(act_path):
    z = np.load(act_path)
    H = {L: z[f"H{L}"] for L in LAYERS}
    N = {L: z[f"N{L}"] for L in EDIT_LAYERS}
    assert len(H[LAYERS[0]]) == len(ctx), "contexts changed since acts.npz was written: delete it"
else:
    H, N = W.collect_acts(m, ctx, layers=LAYERS, neuron_layers=EDIT_LAYERS, K=K_TOKENS)
    np.savez(act_path, **{f"H{L}": v for L, v in H.items()}, **{f"N{L}": v for L, v in N.items()})

# ---- step 4: directions vs Astra's
say("== step 4: directions")
from sklearn.metrics import roc_auc_score  # noqa: E402


def auc(pos, neg):
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])


tr, hd = split == "train", split == "held"
U = {L: W.diff_direction(H[L][tr], kind[tr], "rep", "prod") for L in LAYERS}
U_fin = {L: W.diff_direction(H[L][tr], kind[tr], "rep", "fin") for L in LAYERS}
A = W.astra_directions(exp8, base, "/content")
tr_ids = sorted({c["id"] for c in ctx if c["split"] == "train"})
half = set(rng.choice(tr_ids, len(tr_ids) // 2, replace=False))
hmask = np.array([c["id"] in half for c in ctx])
allA = np.stack(list(A.values()))
top = np.argsort(-(allA ** 2).mean(0))[:20]
keep = np.setdiff1d(np.arange(1024), top)


def nz(v):
    return v[keep] / np.linalg.norm(v[keep])


rows = []
for L in LAYERS:
    a = W.diff_direction(H[L][tr & hmask], kind[tr & hmask])
    b = W.diff_direction(H[L][tr & ~hmask], kind[tr & ~hmask])
    row = dict(layer=L, split_half=abs(a @ b), rep_vs_fin_dir=abs(U[L] @ U_fin[L]),
               cos_p10c=abs(U[L] @ A[("p10c", L)]), cos_p10c_no20=abs(nz(U[L]) @ nz(A[("p10c", L)])),
               auc_held_hand=auc(H[L][hd & (kind == "rep")] @ U[L], H[L][hd & (kind == "prod")] @ U[L]),
               auc_held_p10c=auc(H[L][hd & (kind == "rep")] @ A[("p10c", L)], H[L][hd & (kind == "prod")] @ A[("p10c", L)]))
    if ("p50c", L) in A:
        row["cos_p50c"] = abs(U[L] @ A[("p50c", L)])
        row["cos_p50c_no20"] = abs(nz(U[L]) @ nz(A[("p50c", L)]))
    rows.append(row)
D = pd.DataFrame(rows).set_index("layer")
D.to_csv(f"{OUT}/directions.csv")
np.savez(f"{OUT}/hand_directions.npz", **{f"U{L}": U[L] for L in LAYERS})
say("random |cos| in R^1024: 99.9% below 0.10")
say(D.to_string())

# ---- step 5: edits + teacher-forced check
say("== step 5: edits and log-prob test")


def name(L):
    return f"{W.P}layers.{L}.mlp.down_proj.weight"


W0 = {L: m.get_submodule(f"model.layers.{L}.mlp.down_proj").weight.detach().double().cpu().numpy() for L in EDIT_LAYERS}
act_rep = {L: N[L][tr & (kind == "rep")].mean(0) for L in EDIT_LAYERS}
EDITS, COLS = {}, {}
for a in ALPHAS:
    EDITS[f"hand_a{a:g}"] = {}
    for L in EDIT_LAYERS:
        EDITS[f"hand_a{a:g}"][L], COLS[(a, L)] = W.top_k_edit(W0[L], U[L], act_rep[L], TOPK, a)
EDITS["random_a6"] = {L: W.random_control(W0[L], U[L], act_rep[L], TOPK, 6.0, seed=L)[0] for L in EDIT_LAYERS}
EDITS["astra_p50c"] = {L: np.load(f"/content/astra_edits/p50c/{name(L)}.npy").astype(np.float64) for L in EDIT_LAYERS}
for L in EDIT_LAYERS:
    p50 = np.where(np.abs(EDITS["astra_p50c"][L] - W0[L]).sum(0) > 0)[0]
    say(f"layer {L}: hand top-{TOPK} neurons overlap Astra's 64 in {len(set(COLS[(6.0, L)]) & set(p50))} (chance ~1.1)")
heldc = [c for c in ctx if c["split"] == "held"]
hk = np.array([c["kind"] for c in heldc])
lp_path = f"{OUT}/logprob_test.csv"
if not os.path.exists(lp_path):
    lp0 = W.logprob_test(m, heldc)
    res = {}
    for e, ed in EDITS.items():
        d = W.logprob_test(m, heldc, {L: ed[L] for L in EDIT_LAYERS}) - lp0
        res[e] = {f"dlogp_{k}": d[hk == k].mean() for k in ("rep", "prod", "fin")}
        res[e].update({f"se_{k}": d[hk == k].std() / np.sqrt((hk == k).sum()) for k in ("rep", "prod", "fin")})
        say(e, res[e])
    pd.DataFrame(res).T.to_csv(lp_path)
say(pd.read_csv(lp_path, index_col=0).to_string())

# ---- step 6: generation on the test problems
say("== step 6: generation")
del m
gc.collect()
torch.cuda.empty_cache()
for e in GEN_EDITS:
    ck = W.make_ckpt(base, f"/content/ckpts/{e}", {name(L): EDITS[e][L] for L in EDIT_LAYERS})
    W.generate(ck, problems_path, "think35", f"{OUT}/{e}__think35.jsonl")
    d = W.score_file(f"{OUT}/{e}__think35.jsonl", exp8)
    say(f"RESULT {e}: acc={d.correct.mean():.3f} cut={d.cut.mean():.3f}")
ref = {"base | think35": f"{WORK}/item3/base__think35.jsonl", "astra p50c | think35": f"{WORK}/item3/p50c__think35.jsonl"}
dfs = {k: W.score_file(v, exp8) for k, v in ref.items()}
dfs.update({f"{e} | think35": W.score_file(f"{OUT}/{e}__think35.jsonl", exp8) for e in GEN_EDITS})
S = W.summarize(dfs, "base | think35")
S.to_csv(f"{OUT}/summary.csv")
say(S.to_string())
say("ALL DONE")
