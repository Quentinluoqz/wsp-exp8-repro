"""Writes item3_decoding_baselines.ipynb and item4_hand_direction.ipynb next to this file."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def nb(cells):
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}}}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


SETUP_DRIVE = """
# Results go to Google Drive when it can be mounted (survives the runtime); otherwise to /content/wsp on the VM
# (e.g. in VS Code's Colab extension, where drive.mount may not work): download them before the runtime ends.
USE_DRIVE = True
import os, sys, importlib
WORK = '/content/wsp'
if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        WORK = '/content/drive/MyDrive/wsp'
    except Exception as e:
        print('Google Drive not available here:', type(e).__name__, str(e)[:200])
print('results folder:', WORK)
os.makedirs(WORK, exist_ok=True)
sys.path.insert(0, '/content')
import wsp_colab as W; importlib.reload(W)
"""

HELPER_SRC = open(os.path.join(HERE, "wsp_colab.py")).read()
WRITE_HELPER = "%%writefile /content/wsp_colab.py\n" + HELPER_SRC


# ------------------------------------------------------------------------------------------- item 3
item3 = [
    md("""
# Item 3 - Decoding baselines for Exp 8

**Question.** Exp 8's evaluation runs Qwen3.5-0.8B with **thinking on** (temperature 0.6, presence penalty 1.5,
32,768 tokens). In that setting ~93 % of MATH-500 answers loop until the token cap, and Astra's best edit (p50c)
mostly removes the loops (item 1). But Qwen's model card says that **0.8B runs in non-thinking mode by default** and
that **in thinking mode it "is more prone to entering thinking loops"**. Exp 8's own `evaluate.py` even has a
non-thinking protocol, with the comment "thinking looped at 0.8B". So: how much of the edit's gain does a
different decoding setting already give, with no edit at all?

| condition | checkpoint | setting |
|---|---|---|
| `base \\| think35` | unedited | Exp 8's evaluation (should reproduce ~90 % cut, ~8-14 % acc) |
| `p50c \\| think35` | Astra's 50-pass edit | same (should reproduce the post's gain) |
| `base \\| nothink35` | unedited | model card's non-thinking benchmark setting, 8,192 tokens |
| `p50c \\| nothink35` | Astra's edit | same: does the edit help where the model barely loops? |
| `base \\| think_t1` | unedited | model card's thinking setting for text (temperature 1.0) |
| `base \\| think35_rep` | unedited | think35 + repetition_penalty 1.1 |

Problems: a fixed random 200 GSM8K test + 200 MATH-500 (the post's benchmarks), 1 sample each, graded with the
release's own grader (`exp8_math/code/mathqa.py`: last `\\boxed{}`, math-verify).

**Before you start**
1. Runtime -> Change runtime type -> **A100** (L4 works, ~3x slower; T4 has no bfloat16, not recommended).
2. Nothing to upload: the notebook writes its helper module itself. Results go to `MyDrive/wsp/` if Drive can be
   mounted, else to `/content/wsp` on the VM.
3. Every generation is saved chunk by chunk. If Colab disconnects: re-run the cells up to "Models and
   problems", then the run cell; it continues where it stopped.

Rough cost (A100, not yet measured): a thinking condition writes up to 400 x 32k tokens, about 15-40 min; a
non-thinking one a few minutes. Use `N_GSM = N_MATH = 100` for a first pass.
"""),
    code("""
# vLLM (generation) + the grader's dependency.  A newer vLLM is needed for Qwen3.5.
!pip install -q -U vllm "math-verify[antlr4_13_2]" datasets
# If loading the model later fails with an unknown architecture (qwen3_5), use the nightly the model card names:
# !pip install -q -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
    md("The helper module (the same as `analysis/colab/wsp_colab.py`), written to the VM."),
    code(WRITE_HELPER),
    code(SETUP_DRIVE + "os.makedirs(f'{WORK}/item3', exist_ok=True)\n"),
    md("## Configuration"),
    code("""
N_GSM, N_MATH = 200, 200          # fixed random test subset (saved once to WORK/problems_test.jsonl, shared with item 4)
BASELINE = 'base | think35'
CONDITIONS = [                    # (name, checkpoint, protocol in W.PROTOCOLS)
    ('base | think35',     'base', 'think35'),
    ('p50c | think35',     'p50c', 'think35'),
    ('base | nothink35',   'base', 'nothink35'),
    ('p50c | nothink35',   'p50c', 'nothink35'),
    ('base | think_t1',    'base', 'think_t1'),
    ('base | think35_rep', 'base', 'think35_rep'),
    # optional:
    # ('base | think_pp0', 'base', 'think_pp0'),
    # ('p10c | think35',   'p10c', 'think35'),
    # ('p25c | think35',   'p25c', 'think35'),
]
for k, v in W.PROTOCOLS.items(): print(f'{k:12s}', v)
"""),
    md("## Models and problems\nThe edited checkpoints are rebuilt bit-exactly from the public base weights by the "
       "release's `apply_edit.py` (it checks every tensor's sha256)."),
    code("""
exp8 = W.get_logs()                       # sparse clone of lm_understand-logs/exp8_math
base = W.get_base()                       # Qwen/Qwen3.5-0.8B at the revision Exp 8 edited
CKPT = {'base': base}
for run in sorted({c[1] for c in CONDITIONS} - {'base'}):
    CKPT[run] = W.official_edit_ckpt(run, exp8, base, f'/content/ckpts/{run}')
problems_path = f'{WORK}/problems_test.jsonl'
probs = W.load_test(problems_path, N_GSM, N_MATH)
print(len(probs), 'problems;', sum(p['source'] == 'gsm8k' for p in probs), 'GSM8K')
"""),
    md("## Smoke test (1-2 min)\nEight problems, non-thinking, to check that vLLM loads the model and the grader works."),
    code("""
smoke = '/content/smoke_problems.jsonl'
W._write_jsonl(smoke, probs[:4] + probs[-4:])
if os.path.exists('/content/smoke_out.jsonl'): os.remove('/content/smoke_out.jsonl')
W.generate(base, smoke, 'nothink35', '/content/smoke_out.jsonl')
W.score_file('/content/smoke_out.jsonl', exp8, cache=False)[['id', 'correct', 'pred', 'cut', 'n_tokens']]
"""),
    md("## Run every condition (resumable)"),
    code("""
def out_path(name):
    return f"{WORK}/item3/{name.replace(' | ', '__')}.jsonl"

for name, ck, proto in CONDITIONS:
    print('=' * 20, name)
    W.generate(CKPT[ck], problems_path, proto, out_path(name))
"""),
    md("## Results"),
    code("""
import pandas as pd
pd.options.display.float_format = '{:.3f}'.format
dfs = {name: W.score_file(out_path(name), exp8) for name, _, _ in CONDITIONS if os.path.exists(out_path(name))}
S = W.summarize(dfs, BASELINE)
S.to_csv(f'{WORK}/item3/summary.csv')
S[['n', 'acc', 'se', 'acc_gsm8k', 'acc_math500', 'cut', 'loop', 'mean_tokens', 'acc_if_finished',
   'gain_vs_base', 'gain_se', 'rescued', 'lost']]
"""),
    code("""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
S[['acc_gsm8k', 'acc_math500']].plot.barh(ax=ax[0]); ax[0].set_xlabel('accuracy'); ax[0].invert_yaxis()
kinds = pd.DataFrame({n: d.kind.value_counts(normalize=True) for n, d in dfs.items()}).T.fillna(0)
cols = [c for c in ['finished', 'cut: loop', 'cut: whitespace', 'cut: no loop'] if c in kinds]
kinds[cols].plot.barh(stacked=True, ax=ax[1], color=['#4c9a5b', '#c0504d', '#e3a33b', '#8c8c8c'][:len(cols)])
ax[1].set_xlabel('share of answers'); ax[1].invert_yaxis(); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{WORK}/item3/fig_item3.png', dpi=140)
"""),
    code("""
# Where does each condition's gain over the baseline come from?  (paired by problem)
b = dfs[BASELINE].set_index(['id', 'sample'])
for name, d in dfs.items():
    if name == BASELINE: continue
    j = d.set_index(['id', 'sample']).join(b, rsuffix='_b', how='inner')
    gained = (j.correct == 1) & (j.correct_b == 0)
    print(f'{name:22s} gained {gained.sum():3d}, of which the baseline answer was cut: {(gained & (j.cut_b == 1)).sum():3d};'
          f'  lost {((j.correct == 0) & (j.correct_b == 1)).sum():3d}')
"""),
    md("""
## How to read it
- **`base | nothink35` close to `p50c | think35`**: the edit mostly recovers what the model's default mode already
  gives; the "math gain" is a fix for a thinking-mode failure that the model card documents.
- **`base | think_t1` or `base | think35_rep` already removes most loops**: a sampling setting does the edit's job.
- **`p50c | nothink35` vs `base | nothink35`**: the edit's effect where loops are rare - like the Gemma follow-up,
  expect little.
- `acc_if_finished` and the "gained ... cut" line test item 1's claim on fresh samples.
"""),
]

# ------------------------------------------------------------------------------------------- item 4
item4 = [
    md("""
# Item 4 - The repetition direction, by hand

**Question.** Astra's edits act on one "repetition direction" in the residual stream (item 2). Can we find that
direction **without an LLM**, with the standard interpretability recipe, and does editing along it reproduce p50c?

1. **Practice generations.** The unedited model answers practice problems (GSM8K / MATH *train*, never the test
   sets) with Exp 8's thinking setting, capped at 8,192 tokens (loops start early: median at ~8 % of the answer).
2. **Contexts.** From each looping answer: a prefix that ends inside the loop (`rep`) and a matched prefix of the
   same answer that ends before the loop starts (`prod`). Finished answers give an unmatched control (`fin`).
3. **Activations** (one forward pass per context, transformers): the residual stream after layers 8-22 and the
   MLP neurons of layers 15 and 19, averaged over the last 32 tokens.
4. **Direction** = mean(rep) - mean(prod), per layer. Compare with Astra's p10c / p50c directions, check that it
   separates held-out contexts, check split-half stability.
5. **Edit** with p50c's exact form, `W = W0 - alpha * u (mask * u^T W0)` on the 64 neurons of layers 15 and 19 that
   write most along u, plus a norm-matched random-direction control. First a cheap teacher-forced test on held-out
   contexts (does the edit lower the log-prob of looping continuations only?), then a full generation on the same
   test problems as item 3.

Same setup as item 3 (A100; nothing to upload). It reuses item 3's test problems and, if present,
its `base | think35` and `p50c | think35` generations.
"""),
    code("""
!pip install -q -U vllm "transformers>=5.17" "math-verify[antlr4_13_2]" datasets
# fast kernels for the linear-attention layers (optional: without them transformers falls back to slower PyTorch code)
!pip install -q -U flash-linear-attention
!nvidia-smi --query-gpu=name,memory.total --format=csv
# If an import error mentions torch/transformers versions: Runtime -> Restart session, then continue from the next cell.
"""),
    md("The helper module (the same as `analysis/colab/wsp_colab.py`), written to the VM."),
    code(WRITE_HELPER),
    code(SETUP_DRIVE + "os.makedirs(f'{WORK}/item4', exist_ok=True)\nimport numpy as np, pandas as pd, collections\n"
         "pd.options.display.float_format = '{:.3f}'.format\n"),
    md("## Configuration"),
    code("""
N_PRACTICE_GSM, N_PRACTICE_MATH = 150, 150   # practice problems for building the direction
N_SAMPLES = 2                                # answers per practice problem (more contexts for the same cost)
LAYERS = list(range(8, 23))                  # residual streams to measure (p10c's range)
EDIT_LAYERS = (15, 19)                       # p50c's layers (direction from hidden_states[L+1], neurons of layer L)
K_TOKENS, MAX_CTX = 32, 8000
TOPK, ALPHAS = 64, (3.0, 6.0)                # p50c used top-64 and alpha 6
HELDOUT = 0.2                                # share of practice problems held out for the checks
GEN_EDITS = ['hand_a6', 'random_a6']         # edits to evaluate by generation (add 'hand_a3' if time allows)
"""),
    code("""
exp8 = W.get_logs(); base = W.get_base()
practice_path = f'{WORK}/item4/practice_problems.jsonl'
practice = W.load_practice(practice_path, exp8, N_PRACTICE_GSM, N_PRACTICE_MATH)
problems_path = f'{WORK}/problems_test.jsonl'
test = W.load_test(problems_path)            # the same file as item 3 (created here if item 3 has not run)
print(len(practice), 'practice problems;', len(test), 'test problems')
"""),
    md("## Step 1 - practice generations (unedited model, thinking, 8,192 tokens)"),
    code("""
gen_path = f'{WORK}/item4/practice_base.jsonl'
W.generate(base, practice_path, 'think35_8k', gen_path, n=N_SAMPLES)
sc = W.score_file(gen_path, exp8)
print(sc.kind.value_counts())
print('loop onset (share of the answer): median', sc.onset_rel.median().round(3))
"""),
    md("## Step 2 - contexts"),
    code("""
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(base)
ctx = W.build_contexts(gen_path, sc)
ctx = W.tokenize_contexts(ctx, tok, practice, max_ctx_tokens=MAX_CTX)
rng = np.random.default_rng(0)
ids = sorted({c['id'] for c in ctx}); held = set(rng.choice(ids, int(len(ids) * HELDOUT), replace=False))
for c in ctx: c['split'] = 'held' if c['id'] in held else 'train'
print(pd.crosstab(pd.Series([c['kind'] for c in ctx], name='kind'), pd.Series([c['split'] for c in ctx], name='split')))
print('context length (tokens): median', int(np.median([len(c['ids']) for c in ctx])))
"""),
    md("## Step 3 - activations (cached to Drive)"),
    code("""
import torch
act_path = f'{WORK}/item4/acts.npz'
m, _ = W.load_hf(base)
if os.path.exists(act_path):
    z = np.load(act_path); H = {L: z[f'H{L}'] for L in LAYERS}; N = {L: z[f'N{L}'] for L in EDIT_LAYERS}
else:
    H, N = W.collect_acts(m, ctx, layers=LAYERS, neuron_layers=EDIT_LAYERS, K=K_TOKENS)
    np.savez(act_path, **{f'H{L}': v for L, v in H.items()}, **{f'N{L}': v for L, v in N.items()})
kind = np.array([c['kind'] for c in ctx]); split = np.array([c['split'] for c in ctx])
assert len(kind) == len(H[LAYERS[0]]), 'contexts changed since the cache was written: delete acts.npz'
"""),
    md("## Step 4 - the direction, and how it compares with Astra's"),
    code("""
tr = split == 'train'
U = {L: W.diff_direction(H[L][tr], kind[tr], 'rep', 'prod') for L in LAYERS}
U_fin = {L: W.diff_direction(H[L][tr], kind[tr], 'rep', 'fin') for L in LAYERS}
A = W.astra_directions(exp8, base, '/content')       # {('p10c', L) | ('p50c', 15 / 19): unit vector}

# split-half stability: two disjoint halves of the training problems
tr_ids = sorted({c['id'] for c in ctx if c['split'] == 'train'}); half = set(rng.choice(tr_ids, len(tr_ids) // 2, replace=False))
hmask = np.array([c['id'] in half for c in ctx])
def auc(pos, neg):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])
hd = split == 'held'
rows = []
for L in LAYERS:
    a = W.diff_direction(H[L][tr & hmask], kind[tr & hmask]); b = W.diff_direction(H[L][tr & ~hmask], kind[tr & ~hmask])
    row = dict(layer=L, split_half=abs(a @ b), rep_vs_fin_dir=abs(U[L] @ U_fin[L]),
               cos_p10c=abs(U[L] @ A[('p10c', L)]),
               auc_held=auc(H[L][hd & (kind == 'rep')] @ U[L], H[L][hd & (kind == 'prod')] @ U[L]))
    if ('p50c', L) in A: row['cos_p50c'] = abs(U[L] @ A[('p50c', L)])
    rows.append(row)
D = pd.DataFrame(rows).set_index('layer'); D.to_csv(f'{WORK}/item4/directions.csv')
print('random |cos| in R^1024: 99.9% below 0.10')
D
"""),
    code("""
# Same comparison with the 20 residual dims that dominate Astra's directions removed (item 2's control)
allA = np.stack(list(A.values())); top = np.argsort(-(allA ** 2).mean(0))[:20]; keep = np.setdiff1d(np.arange(1024), top)
nz = lambda v: v[keep] / np.linalg.norm(v[keep])
print('top dims', top[:10].tolist())
print(pd.DataFrame({L: dict(cos_p10c=abs(nz(U[L]) @ nz(A[('p10c', L)])),
                            **({'cos_p50c': abs(nz(U[L]) @ nz(A[('p50c', L)]))} if ('p50c', L) in A else {}))
                    for L in LAYERS}).T)
# does Astra's p50c direction also separate held-out rep / prod contexts?
for L in EDIT_LAYERS:
    print(f'layer {L}: held-out AUC, hand u = {D.auc_held[L]:.3f}, Astra p50c u = '
          f\"{auc(H[L][hd & (kind == 'rep')] @ A[('p50c', L)], H[L][hd & (kind == 'prod')] @ A[('p50c', L)]):.3f}\")
"""),
    md("## Step 5 - edits, and a teacher-forced check on held-out contexts\n"
       "Mean log-prob per token of each held-out context's real next 64 tokens. A good edit lowers it for `rep` "
       "(the loop continuing) and leaves `prod` / `fin` (reasoning continuing) alone."),
    code("""
from safetensors.torch import load_file
name = lambda L: f'{W.P}layers.{L}.mlp.down_proj.weight'
W0 = {L: m.get_submodule(f'model.layers.{L}.mlp.down_proj').weight.detach().double().cpu().numpy() for L in EDIT_LAYERS}
act_rep = {L: N[L][tr & (kind == 'rep')].mean(0) for L in EDIT_LAYERS}
EDITS, COLS = {}, {}
for a in ALPHAS:
    EDITS[f'hand_a{a:g}'] = {}
    for L in EDIT_LAYERS:
        EDITS[f'hand_a{a:g}'][L], COLS[(a, L)] = W.top_k_edit(W0[L], U[L], act_rep[L], TOPK, a)
EDITS['random_a6'] = {L: W.random_control(W0[L], U[L], act_rep[L], TOPK, 6.0, seed=L)[0] for L in EDIT_LAYERS}
EDITS['astra_p50c'] = {L: np.load(f'/content/astra_edits/p50c/{name(L)}.npy').astype(np.float64) for L in EDIT_LAYERS}
p50_cols = {L: np.where(np.abs(EDITS['astra_p50c'][L] - W0[L]).sum(0) > 0)[0] for L in EDIT_LAYERS}
for L in EDIT_LAYERS:
    print(f'layer {L}: hand top-{TOPK} neurons overlap Astra\\'s 64 in {len(set(COLS[(6.0, L)]) & set(p50_cols[L]))} (chance ~1.1)')

heldc = [c for c in ctx if c['split'] == 'held']
hk = np.array([c['kind'] for c in heldc])
lp0 = W.logprob_test(m, heldc)
res = {}
for e, ed in EDITS.items():
    d = W.logprob_test(m, heldc, {L: ed[L] for L in EDIT_LAYERS}) - lp0
    res[e] = {k: d[hk == k].mean() for k in ('rep', 'prod', 'fin')}
LP = pd.DataFrame(res).T; LP.columns = [f'dlogp_{k}' for k in LP.columns]; LP.to_csv(f'{WORK}/item4/logprob_test.csv')
LP   # nats / token; Astra reported -0.10 (rep) vs -0.01 (productive) for its edit
"""),
    md("## Step 6 - generation on the test problems (item 3's setting)"),
    code("""
del m; import gc; gc.collect(); torch.cuda.empty_cache()     # free the GPU for vLLM
def out_path(name): return f'{WORK}/item4/{name}__think35.jsonl'
for e in GEN_EDITS:
    ck = W.make_ckpt(base, f'/content/ckpts/{e}', {name(L): EDITS[e][L] for L in EDIT_LAYERS})
    W.generate(ck, problems_path, 'think35', out_path(e))
# baseline and Astra's edit: reuse item 3's generations, else make them
ref = {'base': f'{WORK}/item3/base__think35.jsonl', 'p50c': f'{WORK}/item3/p50c__think35.jsonl'}
if not os.path.exists(ref['base']): W.generate(base, problems_path, 'think35', ref['base'])
if not os.path.exists(ref['p50c']):
    W.generate(W.official_edit_ckpt('p50c', exp8, base, '/content/ckpts/p50c'), problems_path, 'think35', ref['p50c'])
"""),
    code("""
dfs = {'base | think35': W.score_file(ref['base'], exp8), 'astra p50c | think35': W.score_file(ref['p50c'], exp8)}
dfs.update({f'{e} | think35': W.score_file(out_path(e), exp8) for e in GEN_EDITS})
S = W.summarize(dfs, 'base | think35'); S.to_csv(f'{WORK}/item4/summary.csv')
S[['n', 'acc', 'se', 'acc_gsm8k', 'acc_math500', 'cut', 'loop', 'acc_if_finished', 'gain_vs_base', 'gain_se', 'rescued', 'lost']]
"""),
    md("""
## How to read it
- **Step 4**: `cos_p10c` / `cos_p50c` well above 0.10 (also with the top dims removed) and a held-out AUC near 1
  mean the hand-made direction is the one Astra found. `split_half` near 1 means it is stable.
- **Step 5**: the hand edit should lower `dlogp_rep` much more than `dlogp_prod`; the random control should not.
- **Step 6**: if `hand_a6` gets close to `astra p50c` and `random_a6` stays near the baseline, Astra's edit is
  reproducible with a standard diff-of-means recipe - the LLM automated a known method rather than found
  something new. If the hand edit fails while the direction matches, the difference is in Astra's neuron choice
  or its alpha, which is itself worth a closer look.
"""),
]

for fn, cells in (("item3_decoding_baselines.ipynb", item3), ("item4_hand_direction.ipynb", item4)):
    json.dump(nb(cells), open(os.path.join(HERE, fn), "w"), indent=1, ensure_ascii=False)
    print("wrote", fn, len(cells), "cells")
