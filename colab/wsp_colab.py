"""Helpers for items 3 and 4 (Colab).  Put this file next to the notebooks in your Drive folder.

Item 3: decoding baselines - is Exp 8's gain something a sampling setting / non-thinking mode already gives?
Item 4: derive the "repetition direction" by hand (diff of means, no LLM), compare it with Astra's, and test edits.

Generation runs in a subprocess (`python wsp_colab.py gen ...`) so that vLLM releases the GPU after each checkpoint.
Every result is written to the Drive folder and every step is resumable.
"""
import argparse
import gc
import glob
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np

BASE_REPO = "Qwen/Qwen3.5-0.8B"
BASE_REV = "2fc06364715b967f1860aea9cf38778875588b17"          # the revision Exp 8 edited
LOGS_URL = "https://github.com/RikiyaT/lm_understand-logs.git"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
P = "model.language_model."                                      # tensor-name prefix in the checkpoint

# vLLM sampling settings.  think35 / nothink35 are exactly Exp 8's evaluate.py SAMPLING_THINK35 / SAMPLING_NOTHINK35.
PROTOCOLS = {
    # Exp 8's evaluation: thinking on, the model card's thinking benchmark setting, 32,768 tokens
    "think35": dict(thinking=True, temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=32768),
    # model card, "thinking mode for text tasks"
    "think_t1": dict(thinking=True, temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=32768),
    # think35 + a multiplicative repetition penalty
    "think35_rep": dict(thinking=True, temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5,
                        repetition_penalty=1.1, max_tokens=32768),
    # think35 without the presence penalty (model card: "precise coding" setting)
    "think_pp0": dict(thinking=True, temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0.0, max_tokens=32768),
    # the model card's NON-thinking benchmark setting; non-thinking is Qwen3.5-0.8B's default mode
    "nothink35": dict(thinking=False, temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, max_tokens=8192),
    # item 4: practice generations only need the start of each answer (loops start early)
    "think35_8k": dict(thinking=True, temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=8192),
}


# ===================================================================== setup
q = shlex.quote


def sh(cmd):
    """Run a shell command and stream its output into the notebook cell (a plain subprocess.run would send it to the
    kernel's own terminal, so a long generation would look frozen)."""
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait():
        raise subprocess.CalledProcessError(p.returncode, cmd)


def get_logs(dest="/content/lm_understand-logs"):
    """Sparse clone of the logs repo: exp8_math only (code, edits, practice problems)."""
    if not os.path.exists(os.path.join(dest, "exp8_math/code/apply_edit.py")):
        sh(f"git clone --depth 1 --filter=blob:none --sparse {LOGS_URL} {q(dest)}")
        sh(f"cd {q(dest)} && git sparse-checkout set exp8_math")
    return os.path.join(dest, "exp8_math")


def get_base(dest="/content/models/Qwen3.5-0.8B"):
    from huggingface_hub import snapshot_download
    snapshot_download(BASE_REPO, revision=BASE_REV, local_dir=dest)
    return dest


def base_st(base):
    return glob.glob(os.path.join(base, "model*.safetensors"))[0]


def _copy_non_weights(base, out):
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(base):
        src = os.path.join(base, f)
        if os.path.isfile(src) and not f.endswith(".safetensors"):
            shutil.copy(src, os.path.join(out, f))


def official_edit_ckpt(run, exp8, base, out):
    """A full checkpoint with Astra's evaluated edit of `run` (p10c / p25c / p50c), rebuilt bit-exactly and
    hash-checked by the release's apply_edit.py."""
    st = base_st(base)
    target = os.path.join(out, os.path.basename(st))
    if not os.path.exists(os.path.join(out, "DONE")):
        _copy_non_weights(base, out)
        sh(f"{q(sys.executable)} {q(exp8 + '/code/apply_edit.py')} {q(st)} {q(exp8 + '/runs/' + run + '/edit')} {q(target)}")
        open(os.path.join(out, "DONE"), "w").write(run)
    return out


def make_ckpt(base, out, replace):
    """A full checkpoint = base with tensors replaced.  replace: {full tensor name: float array}; stored as bf16."""
    import torch
    from safetensors.torch import load_file, save_file
    if os.path.exists(os.path.join(out, "DONE")):
        return out
    _copy_non_weights(base, out)
    st = base_st(base)
    T = load_file(st)
    for k, v in replace.items():
        assert k in T and tuple(T[k].shape) == tuple(v.shape), k
        T[k] = torch.as_tensor(np.asarray(v, dtype=np.float32)).to(torch.bfloat16).contiguous()
    save_file(T, os.path.join(out, os.path.basename(st)), metadata={"format": "pt"})
    json.dump(sorted(replace), open(os.path.join(out, "DONE"), "w"))
    del T
    gc.collect()
    return out


# ===================================================================== grading (the release's own grader)
_GRADER = []


def grader(exp8):
    """exp8_math/code/mathqa.py: extract() = the last \\boxed{} after </think>, is_correct() = math-verify."""
    if not _GRADER:
        stub = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_stubs")
        os.makedirs(stub, exist_ok=True)
        open(os.path.join(stub, "mcqa.py"), "w").write(
            "def loop_signals(text):\n    return {'repetitive_end': False}\n")   # only used by a function we skip
        for p in (stub, os.path.join(exp8, "code")):
            if p not in sys.path:
                sys.path.insert(0, p)
        import mathqa
        _GRADER.append(mathqa)
    return _GRADER[0]


# ===================================================================== problems
def _write_jsonl(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def read_jsonl(path):
    return [json.loads(l) for l in open(path)]


def load_test(path, n_gsm=200, n_math=200, seed=0):
    """A fixed random subset of GSM8K test and MATH-500 (the benchmarks of the post), saved once to `path`."""
    if os.path.exists(path):
        return read_jsonl(path)
    from datasets import load_dataset
    rng = random.Random(seed)
    g = load_dataset("openai/gsm8k", "main", split="test")
    m = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = []
    for i in sorted(rng.sample(range(len(g)), n_gsm)):
        rows.append(dict(id=f"gsm8k_test_{i}", source="gsm8k", problem=g[i]["question"],
                         answer=g[i]["answer"].split("####")[-1].strip().replace(",", "")))
    for i in sorted(rng.sample(range(len(m)), n_math)):
        rows.append(dict(id=m[i]["unique_id"], source="math500", problem=m[i]["problem"], answer=m[i]["answer"]))
    _write_jsonl(path, rows)
    return rows


def load_practice(path, exp8, n_gsm=150, n_math=150, seed=1):
    """Practice problems Astra could use (GSM8K train, MATH train) - never the test sets."""
    if os.path.exists(path):
        return read_jsonl(path)
    rng = random.Random(seed)
    rows = []
    for name, n in (("gsm8k_train", n_gsm), ("math_train", n_math)):
        allr = read_jsonl(os.path.join(exp8, "practice", f"{name}.jsonl"))
        rows += [dict(id=r["id"], source=r["source"], problem=r["problem"], answer=r["answer"]) for r in rng.sample(allr, n)]
    _write_jsonl(path, rows)
    return rows


def prompt_text(tok, problem, thinking):
    return tok.apply_chat_template([{"role": "user", "content": problem + "\n" + INSTRUCTION}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=thinking)


# ===================================================================== generation (vLLM, in a subprocess)
def generate(ckpt, problems_path, protocol, out_path, n=1, gpu_mem=0.85, chunk=50):
    """Resumable: answers are appended chunk by chunk to out_path; problems already there are skipped."""
    done = {r["id"] for r in read_jsonl(out_path)} if os.path.exists(out_path) else set()
    todo = [r for r in read_jsonl(problems_path) if r["id"] not in done]
    if not todo:
        print(f"[skip] {out_path}: complete")
        return out_path
    t = time.time()
    sh(f"{q(sys.executable)} {q(os.path.abspath(__file__))} gen --ckpt {q(ckpt)} --problems {q(problems_path)} "
       f"--protocol {protocol} --out {q(out_path)} --n {n} --gpu_mem {gpu_mem} --chunk {chunk}")
    print(f"[done] {out_path} in {(time.time() - t) / 60:.1f} min")
    return out_path


def _gen_main(a):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    cfg = dict(PROTOCOLS[a.protocol])
    thinking = cfg.pop("thinking")
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    done = {r["id"] for r in read_jsonl(a.out)} if os.path.exists(a.out) else set()
    rows = [r for r in read_jsonl(a.problems) if r["id"] not in done]
    llm = None
    common = dict(model=a.ckpt, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem,
                  max_model_len=cfg["max_tokens"] + 2048, seed=0)
    for extra in ({"language_model_only": True}, {"limit_mm_per_prompt": {"image": 0, "video": 0}}, {}):
        try:
            llm = LLM(**common, **extra)
            print("vLLM started with", extra or "no extra options", flush=True)
            break
        except (TypeError, ValueError) as e:      # option unknown to this vLLM version
            print("vLLM option", extra, "refused:", str(e)[:200], flush=True)
    if llm is None:
        raise RuntimeError("vLLM could not load the model; install a newer vLLM (see the install cell)")
    sp = SamplingParams(n=a.n, seed=0, **cfg)
    for s in range(0, len(rows), a.chunk):
        part = rows[s:s + a.chunk]
        t0 = time.time()
        outs = llm.generate([prompt_text(tok, r["problem"], thinking) for r in part], sp, use_tqdm=False)
        with open(a.out, "a") as f:
            for r, o in zip(part, outs):
                for k, x in enumerate(o.outputs):
                    f.write(json.dumps(dict(id=r["id"], source=r["source"], answer=r["answer"], sample=k,
                                            text=x.text, finish=x.finish_reason, n_tokens=len(x.token_ids))) + "\n")
        ntok = sum(len(x.token_ids) for o in outs for x in o.outputs)
        ncut = sum(x.finish_reason == "length" for o in outs for x in o.outputs)
        print(f"  {min(s + a.chunk, len(rows))}/{len(rows)} problems | this chunk: {time.time() - t0:.0f}s, "
              f"{ntok / max(time.time() - t0, 1e-9):.0f} tok/s, {ncut}/{sum(len(o.outputs) for o in outs)} cut", flush=True)


# ===================================================================== scoring
NGRAM = 8


def loop_stats(text):
    """Word-level loop measures (same as item 1).
    dup_tail: share of the last 500 word-8-grams seen earlier; onset_char: where the text turns into a loop (the
    earliest word from which >= 90% of 8-grams are repeats, the loop lasting >= 200 words), -1 if never;
    onset_rel: the same as a share of the words; period: words between the last 8-gram and its previous occurrence."""
    spans = [m.start() for m in re.finditer(r"\S+", text)]
    w = text.split()
    n = len(w) - NGRAM + 1
    tail = text[-2000:]
    ws_tail = len(tail) > 0 and sum(c.isspace() for c in tail) / len(tail) >= 0.9
    if n <= 0:
        return dict(words=len(w), dup_tail=0.0, onset_char=-1, onset_rel=np.nan, period=np.nan, ws_tail=ws_tail)
    seen, last = set(), {}
    dup = np.zeros(n, dtype=np.int8)
    dists = []
    for i in range(n):
        g = tuple(w[i:i + NGRAM])
        if g in seen:
            dup[i] = 1
            dists.append((i, i - last[g]))
        seen.add(g)
        last[g] = i
    suf = np.cumsum(dup[::-1])[::-1] / np.arange(n, 0, -1)
    ok = np.where((suf >= 0.9) & (np.arange(n) < n - 200))[0]
    onset = int(ok[0]) if len(ok) else -1
    after = [d for i, d in dists if onset >= 0 and i >= onset]
    return dict(words=len(w), dup_tail=float(dup[-500:].mean()), onset_char=spans[onset] if onset >= 0 else -1,
                onset_rel=onset / n if onset >= 0 else np.nan,
                period=float(np.median(after)) if after else np.nan, ws_tail=ws_tail)


def score_file(path, exp8, cache=True):
    """One row per answer: correct, cut (finish_reason == length), loop measures, kind."""
    import pandas as pd
    cp = path.replace(".jsonl", ".scored.csv")
    if cache and os.path.exists(cp) and os.path.getmtime(cp) > os.path.getmtime(path):
        return pd.read_csv(cp)
    Q = grader(exp8)
    recs = []
    for r in read_jsonl(path):
        pred, nbox = Q.extract(r["text"])
        st = loop_stats(r["text"])
        cut = r["finish"] == "length"
        kind = "finished" if not cut else "cut: whitespace" if st["ws_tail"] else \
            "cut: loop" if st["dup_tail"] >= 0.8 else "cut: no loop"
        recs.append(dict(id=r["id"], source=r["source"], sample=r["sample"], correct=int(Q.is_correct(pred, r["answer"])),
                         pred=pred, cut=int(cut), n_tokens=r["n_tokens"], closed_think=int("</think>" in r["text"]),
                         kind=kind, **st))
    df = pd.DataFrame(recs)
    if cache:
        df.to_csv(cp, index=False)
    return df


def summarize(dfs, baseline):
    """dfs: {condition: scored DataFrame}.  Accuracy with SE, cut / loop shares, tokens, paired gain vs baseline."""
    import pandas as pd
    b = dfs[baseline].set_index(["id", "sample"])["correct"]
    rows = []
    for name, d in dfs.items():
        x = d.set_index(["id", "sample"])["correct"]
        j = pd.concat([x, b], axis=1, keys=["c", "b"]).dropna()
        diff = j.c - j.b
        row = dict(condition=name, n=len(d), acc=d.correct.mean(), se=d.correct.std() / np.sqrt(len(d)),
                   cut=d.cut.mean(), loop=(d.kind == "cut: loop").mean(), mean_tokens=d.n_tokens.mean(),
                   gain_vs_base=diff.mean(), gain_se=diff.std() / np.sqrt(len(diff)) if len(diff) > 1 else np.nan,
                   rescued=int(((j.c == 1) & (j.b == 0)).sum()), lost=int(((j.c == 0) & (j.b == 1)).sum()))
        for s in sorted(d.source.unique()):
            row[f"acc_{s}"] = d[d.source == s].correct.mean()
        row["acc_if_finished"] = d[d.cut == 0].correct.mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("condition")


# ===================================================================== item 4: contexts and activations
def build_contexts(gen_path, scored, seed=0):
    """From base-model practice generations: contexts that end INSIDE a loop ("rep") and matched contexts from the
    same answers that end well before the loop starts ("prod"); plus contexts from answers that finished
    ("fin", an unmatched productive control).  Each keeps the next 64 tokens of the real text (for the log-prob test)."""
    rng = random.Random(seed)
    gens = {(r["id"], r["sample"]): r for r in read_jsonl(gen_path)}
    ctx = []
    for _, s in scored.iterrows():
        g = gens[(s["id"], s["sample"])]
        t = g["text"]
        if s["kind"] == "cut: loop" and s["onset_char"] >= 1500:
            o = int(s["onset_char"])
            per = s["period"] if s["period"] == s["period"] else 20
            into = min(len(t) - 400, o + max(300, int(3 * per * 6)))       # ~3 loop periods (~6 chars / word)
            prod = int(o * rng.uniform(0.3, 0.8))
            for kind, end in (("rep", into), ("prod", prod)):
                ctx.append(dict(kind=kind, gid=f"{s['id']}#{s['sample']}", id=s["id"], prefix=t[:end],
                                cont=t[end:end + 600]))
        elif s["kind"] == "finished" and s["closed_think"]:
            close = t.find("</think>")
            if close > 1500:
                end = int(close * rng.uniform(0.2, 0.7))
                ctx.append(dict(kind="fin", gid=f"{s['id']}#{s['sample']}", id=s["id"], prefix=t[:end],
                                cont=t[end:end + 600]))
    return ctx


def tokenize_contexts(ctx, tok, problems, thinking=True, max_ctx_tokens=8000, cont_tokens=64):
    out = []
    pmap = {p["id"]: p for p in problems}
    for c in ctx:
        head = prompt_text(tok, pmap[c["id"]]["problem"], thinking)
        ids = tok(head + c["prefix"], add_special_tokens=False).input_ids
        cont = tok(c["cont"], add_special_tokens=False).input_ids[:cont_tokens]
        if len(ids) <= max_ctx_tokens and len(cont) == cont_tokens:
            out.append(dict(c, ids=ids, cont_ids=cont))
    return out


def load_hf(base, device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    return m.to(device).eval(), tok


def collect_acts(model, ctxs, layers=tuple(range(8, 23)), neuron_layers=(15, 19), K=32, log_every=50):
    """Per context: residual stream after each layer in `layers` (hidden_states[L+1]) and the MLP neuron
    activations (input of down_proj) of `neuron_layers`, both averaged over the context's last K tokens."""
    import torch
    dev = next(model.parameters()).device
    store = {}
    hooks = [model.get_submodule(f"model.layers.{L}.mlp.down_proj").register_forward_hook(
        lambda mod, inp, out, L=L: store.__setitem__(L, inp[0][0, -K:].float().mean(0).cpu().numpy()))
        for L in neuron_layers]
    H = {L: [] for L in layers}
    N = {L: [] for L in neuron_layers}
    t = time.time()
    try:
        with torch.no_grad():
            for i, c in enumerate(ctxs):
                o = model(torch.tensor([c["ids"]], device=dev), output_hidden_states=True, use_cache=False)
                for L in layers:
                    H[L].append(o.hidden_states[L + 1][0, -K:].float().mean(0).cpu().numpy())
                for L in neuron_layers:
                    N[L].append(store[L])
                del o
                if (i + 1) % log_every == 0:
                    print(f"  {i + 1}/{len(ctxs)} contexts, {time.time() - t:.0f}s", flush=True)
    finally:
        for h in hooks:
            h.remove()
    return {L: np.stack(v) for L, v in H.items()}, {L: np.stack(v) for L, v in N.items()}


def diff_direction(X, kinds, pos="rep", neg="prod"):
    kinds = np.asarray(kinds)
    d = X[kinds == pos].mean(0) - X[kinds == neg].mean(0)
    return d / np.linalg.norm(d)


def astra_directions(exp8, base, work):
    """Residual directions of Astra's p10c (one per layer, 8-22) and p50c (layers 15, 19) edits: top left singular
    vector of (edited - base) for each tensor (rank one, see item 2)."""
    import torch
    from safetensors.torch import load_file
    out = {}
    W0 = None
    for run in ("p10c", "p50c"):
        d = os.path.join(work, "astra_edits", run)
        if not os.path.exists(d):
            sh(f"{q(sys.executable)} {q(exp8 + '/code/apply_edit.py')} {q(base_st(base))} {q(exp8 + '/runs/' + run + '/edit')} {q(d)}")
        if W0 is None:
            W0 = load_file(base_st(base))
        for f in sorted(glob.glob(f"{d}/*.npy")):
            n = os.path.basename(f)[:-4]
            if run == "p10c" and "mlp.down_proj" not in n:
                continue                                      # p10c: one direction per layer, shared by its tensors
            D = np.load(f).astype(np.float64) - W0[n].float().numpy().astype(np.float64)
            u = np.linalg.svd(D, full_matrices=False)[0][:, 0]
            out[(run, int(n.split("layers.")[1].split(".")[0]))] = u
    return out


def top_k_edit(W0, u, act_mean, k=64, alpha=6.0):
    """Astra's p50c form: W = W0 - alpha * u (mask * (u^T W0)), on the k neurons that write the most along u
    in the measured contexts (score_j = |mean activation_j * u . W0[:, j]|).  W0: (1024, 3584) float64."""
    w = u @ W0
    score = np.abs(act_mean * w)
    cols = np.argsort(-score)[:k]
    mask = np.zeros_like(w)
    mask[cols] = 1
    return W0 - alpha * np.outer(u, mask * w), cols


def random_control(W0, u, act_mean, k, alpha, seed=0):
    """A random unit direction orthogonal to u, same selection rule, scaled so that |dW| matches the u-edit's."""
    r = np.random.default_rng(seed).standard_normal(len(u))
    r -= (r @ u) * u
    r /= np.linalg.norm(r)
    We, _ = top_k_edit(W0, u, act_mean, k, alpha)
    Wr, cols = top_k_edit(W0, r, act_mean, k, 1.0)
    s = np.linalg.norm(We - W0) / np.linalg.norm(Wr - W0)
    return W0 + s * (Wr - W0), cols, r


def logprob_test(model, ctxs, edits=None):
    """Mean log-prob per token of each context's real 64-token continuation, with optional in-memory edits
    {layer: down_proj W (1024, 3584)}.  Teacher-forced: one forward pass per context."""
    import torch
    dev = next(model.parameters()).device
    saved = {}
    try:
        for L, W in (edits or {}).items():
            p = model.get_submodule(f"model.layers.{L}.mlp.down_proj").weight
            saved[L] = p.data.clone()
            p.data = torch.as_tensor(W, dtype=p.dtype, device=dev)
        res = []
        with torch.no_grad():
            for c in ctxs:
                ids = torch.tensor([c["ids"] + c["cont_ids"]], device=dev)
                lp = torch.log_softmax(model(ids, use_cache=False).logits[0, len(c["ids"]) - 1:-1].float(), -1)
                res.append(lp.gather(1, ids[0, len(c["ids"]):, None]).mean().item())
        return np.array(res)
    finally:
        for L, W in saved.items():
            model.get_submodule(f"model.layers.{L}.mlp.down_proj").weight.data = W


# ===================================================================== CLI (the generation subprocess)
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen"])
    ap.add_argument("--ckpt")
    ap.add_argument("--problems")
    ap.add_argument("--protocol")
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--chunk", type=int, default=50)
    _gen_main(ap.parse_args())
