"""Item 2: what directions did Astra's three Exp-8 edits act on, and are they the same?

    python analyze_edits.py <base model.safetensors> <tokenizer.json> <edited dir> <exp8 code dir> <out dir>

<edited dir>/<run>/<tensor>.npy are the evaluated tensors rebuilt by exp8_math/code/apply_edit.py (hash-checked).
Every number here is linear algebra on weights only: no forward pass.
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

P = "model.language_model."
RNG = np.random.default_rng(0)


def short(name):
    s = name.replace(P, "").replace(".weight", "")
    return s.replace("layers.", "L").replace(".mlp.down_proj", " mlp").replace(".linear_attn.out_proj", " lin") \
            .replace(".self_attn.o_proj", " attn")


def layer_of(name):
    return int(name.split("layers.")[1].split(".")[0])


def main(base_path, tok_path, edited, code_dir, out):
    sys.path.insert(0, code_dir)
    import st_numpy as S
    from tokenizers import Tokenizer

    os.makedirs(out, exist_ok=True)
    rep = []
    say = lambda *a: (print(*a), rep.append(" ".join(str(x) for x in a)))  # noqa: E731

    runs = {r: sorted(glob.glob(f"{edited}/{r}/*.npy")) for r in ["p10c", "p25c", "p50c"]}
    names = sorted({os.path.basename(f)[:-4] for fs in runs.values() for f in fs})
    extra = [P + "embed_tokens.weight", P + "norm.weight"] + \
            [P + f"layers.{l}.mlp.down_proj.weight" for l in range(24)] + \
            [P + f"layers.{l}.self_attn.o_proj.weight" for l in (3, 7, 11, 15, 19, 23)]
    base = {k: S.bf16_to_f32(b).astype(np.float64) for k, (dt, b) in S.load(base_path, sorted(set(names + extra))).items()}

    # ------------------------------------------------------------------ 1. directions from the deltas
    say("## 1. Each edited tensor: rank of the change and its residual-stream direction u")
    dirs = {}            # (run, tensor) -> unit u (1024,)
    for run, files in runs.items():
        for f in files:
            n = os.path.basename(f)[:-4]
            W0, W1 = base[n], np.load(f).astype(np.float64)
            D = W1 - W0
            U, s, Vt = np.linalg.svd(D, full_matrices=False)
            u = U[:, 0] * np.sign(U[np.argmax(np.abs(U[:, 0])), 0])
            dirs[(run, n)] = u
            cols = np.where(np.abs(D).sum(0) > 0)[0]
            say(f"{run:5s} {short(n):12s} |D|/|W0|={np.linalg.norm(D) / np.linalg.norm(W0):.3f}  "
                f"s2/s1={s[1] / s[0]:.4f}  changed cols={len(cols)}")

    # p10c stores its rank-one factors: check they match
    z = np.load(os.path.join(os.path.dirname(code_dir), "runs/p10c/edit/edit.npz"))
    man = json.load(open(os.path.join(os.path.dirname(code_dir), "runs/p10c/edit/manifest.json")))
    c = [abs(np.dot(z[f"t{i}_a"][:, 0] / np.linalg.norm(z[f"t{i}_a"]), dirs[("p10c", t["name"])]))
         for i, t in enumerate(man["tensors"])]
    say(f"p10c: |cos(stored factor a, SVD u)| min={min(c):.6f}")

    # ------------------------------------------------------------------ 2. are the directions the same?
    say("\n## 2. Cosine similarity between directions (|cos|; random pair in R^1024: ~0.025, 99.9% < 0.10)")
    p10 = {layer_of(n): {} for (r, n) in dirs if r == "p10c"}
    for (r, n), u in dirs.items():
        if r == "p10c":
            p10[layer_of(n)]["lin" if "linear" in n else "attn" if "self_attn" in n else "mlp"] = u
    keys = [k for k in dirs if k[0] == "p10c"]
    keys.sort(key=lambda k: (layer_of(k[1]), "mlp" in k[1]))
    keys += [k for k in dirs if k[0] == "p50c"]
    V = np.stack([dirs[k] for k in keys])
    C = np.abs(V @ V.T)
    p10_idx = [i for i, k in enumerate(keys) if k[0] == "p10c"]
    off = C[np.ix_(p10_idx, p10_idx)][~np.eye(len(p10_idx), dtype=bool)]
    say(f"p10c among its 30 tensors: |cos| min={off.min():.3f} median={np.median(off):.3f} max={off.max():.3f}")
    sub = C[np.ix_(p10_idx, p10_idx)]
    reps = []
    for i in range(len(p10_idx)):
        if not any(sub[i, j] > 0.9999 for j in reps):
            reps.append(i)
    say(f"p10c distinct directions (|cos|>0.9999 merged): {len(reps)}  (one per layer, shared by its attn/MLP outputs)")
    adj = [C[p10_idx[reps[i]], p10_idx[reps[i + 1]]] for i in range(len(reps) - 1)]
    say(f"p10c adjacent-layer |cos|: {np.round(adj, 2).tolist()};  L8 vs L22: {C[p10_idx[reps[0]], p10_idx[reps[-1]]]:.2f}")
    for k in [k for k in keys if k[0] == "p50c"]:
        i = keys.index(k)
        row = C[i, p10_idx]
        same_layer = [C[i, j] for j in p10_idx if layer_of(keys[j][1]) == layer_of(k[1])]
        say(f"p50c {short(k[1])}: vs p10c max={row.max():.3f} (at {short(keys[p10_idx[int(row.argmax())]][1])}), "
            f"median={np.median(row):.3f}, same layer={np.round(same_layer, 3).tolist()}")
    i15, i19 = keys.index(("p50c", P + "layers.15.mlp.down_proj.weight")), keys.index(("p50c", P + "layers.19.mlp.down_proj.weight"))
    say(f"p50c u15 vs u19: {C[i15, i19]:.3f}")
    rnd = np.abs(RNG.standard_normal((20000, 1024)) @ RNG.standard_normal((1024,)))
    rnd /= np.linalg.norm(RNG.standard_normal((1024,))) * np.sqrt(1024)
    # proper null: cos of two random unit vectors
    a = RNG.standard_normal((20000, 1024)); b = RNG.standard_normal((20000, 1024))
    null = np.abs((a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1))
    say(f"null |cos| (random unit vectors): mean={null.mean():.3f}, 99.9%={np.quantile(null, .999):.3f}")

    # anisotropy control: how concentrated is each u on a few coordinates?
    say("\n## 2b. Concentration of u on few residual coordinates (massive-activation dims could fake similarity)")
    for k in keys[:: max(1, len(keys) // 6)] + [keys[i15], keys[i19]]:
        u = dirs[k]
        top = np.argsort(-np.abs(u))[:5]
        say(f"{k[0]} {short(k[1]):12s} participation ratio={1 / np.sum(u ** 4):7.1f} (uniform=1024)  "
            f"top-5 dims {top.tolist()} carry {np.sum(u[top] ** 2):.1%} of |u|^2")

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(C, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(keys)), [f"{k[0][:3]} {short(k[1])}" for k in keys], rotation=90, fontsize=6)
    ax.set_yticks(range(len(keys)), [f"{k[0][:3]} {short(k[1])}" for k in keys], fontsize=6)
    plt.colorbar(im, ax=ax, label="|cos|")
    ax.set_title("Directions of the p10c (30 tensors) and p50c (2 tensors) edits")
    fig.tight_layout()
    fig.savefig(f"{out}/fig1_direction_cosines.png", dpi=150)

    # ------------------------------------------------------------------ 3. p50c: which neurons, and why
    say("\n## 3. p50c: the 64 MLP neurons per layer; formula check; overlap with p10c's direction")
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
    for ax, L in zip(axs, (15, 19)):
        n = P + f"layers.{L}.mlp.down_proj.weight"
        W0 = base[n]
        D = np.load(f"{edited}/p50c/{n}.npy").astype(np.float64) - W0
        u = dirs[("p50c", n)]
        cols = np.where(np.abs(D).sum(0) > 0)[0]
        w = u @ W0                                   # how much each neuron writes along u
        alpha = -(u @ D[:, cols]) / w[cols]          # D[:,j] = -alpha * u * w_j  => alpha
        resid = np.linalg.norm(D[:, cols] + np.outer(u, alpha.mean() * w[cols])) / np.linalg.norm(D[:, cols])
        rank_sel = np.argsort(np.argsort(-np.abs(w)))[cols]
        say(f"L{L}: alpha median={np.median(alpha):.2f} (Astra: 6)  residual of rank-1 fit={resid:.3f} (bf16 rounding)")
        say(f"     selected neurons' rank by |u.W0[:,j]| among 3584: median={np.median(rank_sel):.0f}, max={rank_sel.max()}"
            f"  -> {np.mean(rank_sel < 64):.0%} are the top-64 writers of u")
        say(f"     share of the layer's total |u.W0|^2 written by the 64: {np.sum(w[cols] ** 2) / np.sum(w ** 2):.1%}")
        say(f"     the layer writes along u {np.linalg.norm(w) / (np.linalg.norm(W0) / np.sqrt(1024)):.2f}x a random direction's amount")
        # same question for p10c's direction in that layer's MLP
        u10 = p10[L]["mlp"]
        w10 = u10 @ W0
        top10 = np.argsort(-np.abs(w10))[:64]
        say(f"     p10c's L{L} mlp direction: its top-64 writers overlap p50c's 64 in {len(set(top10) & set(cols))} neurons"
            f" (chance ~{64 * 64 / 3584:.1f})")
        ax.semilogy(np.sort(np.abs(w))[::-1], color="#888", label="all neurons")
        ax.semilogy(np.sort(rank_sel), np.sort(np.abs(w[cols]))[::-1], "o", ms=3, color="#c0504d", label="p50c's 64")
        ax.set_title(f"layer {L}: |u . W_down[:, j]| sorted")
        ax.set_xlabel("neuron rank")
        ax.set_xlim(0, 400)
        ax.legend()
        np.savetxt(f"{out}/p50c_L{L}_neurons.txt", cols, fmt="%d")
    fig.tight_layout()
    fig.savefig(f"{out}/fig2_p50c_neurons.png", dpi=150)

    # ------------------------------------------------------------------ 4. p25c: the head it switched off
    say("\n## 4. p25c zeroed head 7 of layer 15. Does that head write along the 'loop' directions?")
    n = P + "layers.15.self_attn.o_proj.weight"
    Wo = base[n]
    E = np.load(f"{edited}/p25c/{n}.npy")
    zero_cols = np.where(np.all(E == 0, axis=0))[0]
    say(f"zeroed columns: {zero_cols.min()}..{zero_cols.max()} ({len(zero_cols)}) -> head {zero_cols.min() // 256}")
    refs = {"p50c u15": dirs[("p50c", P + "layers.15.mlp.down_proj.weight")],
            "p50c u19": dirs[("p50c", P + "layers.19.mlp.down_proj.weight")],
            "p10c L15 attn": p10[15]["attn"]}
    heads = np.arange(8)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for j, (lab, u) in enumerate(refs.items()):
        # fraction of the head's output energy that lands on u, relative to the 1/1024 of a random direction
        frac = [np.sum((u @ Wo[:, h * 256:(h + 1) * 256]) ** 2) / np.sum(Wo[:, h * 256:(h + 1) * 256] ** 2) * 1024 for h in heads]
        say(f"{lab:14s} per-head alignment (x random): " + " ".join(f"h{h}={f:.2f}" for h, f in zip(heads, frac)))
        ax.bar(heads + (j - 1) * 0.27, frac, width=0.27, label=lab)
    ax.axhline(1, color="k", lw=0.6, ls="--")
    ax.set_xlabel("layer-15 attention head")
    ax.set_ylabel("output energy on u (x random)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{out}/fig3_p25c_head.png", dpi=150)

    # ------------------------------------------------------------------ 5. logit lens of the directions
    say("\n## 5. Logit lens: tokens most promoted at each end of u (u * final-norm weight . embedding)")
    tok = Tokenizer.from_file(tok_path)
    Emb = base[P + "embed_tokens.weight"]
    g = 1.0 + base[P + "norm.weight"]  # Qwen3.5 RMSNorm uses (1 + weight); see check below
    say(f"final norm weight: mean={base[P + 'norm.weight'].mean():.3f} (near 0 -> (1+w) convention, near 1 -> w)")
    if base[P + "norm.weight"].mean() > 0.5:
        g = base[P + "norm.weight"]
    lens = {"p50c u15": refs["p50c u15"], "p50c u19": refs["p50c u19"],
            "p10c L15 mlp": p10[15]["mlp"], "p10c L19 mlp": p10[19]["mlp"], "p10c L22 mlp": p10[22]["mlp"]}
    with open(f"{out}/logit_lens.md", "w") as fh:
        for lab, u in lens.items():
            sc = Emb @ (u * g)
            sc = (sc - sc.mean()) / sc.std()
            o = np.argsort(sc)
            dec = lambda ids: [repr(tok.decode([int(i)])) for i in ids]  # noqa: E731
            pos, neg = dec(o[::-1][:25]), dec(o[:25])
            fh.write(f"### {lab}\n\n+u: {' '.join(pos)}\n\n-u: {' '.join(neg)}\n\n")
            say(f"{lab}\n   +u: {' '.join(pos[:15])}\n   -u: {' '.join(neg[:15])}")
            say(f"   max |z| = {np.abs(sc).max():.1f}")

    open(f"{out}/report.txt", "w").write("\n".join(rep) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
