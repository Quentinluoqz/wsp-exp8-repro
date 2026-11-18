"""Control for item 2: are the p10c / p50c / p25c directions similar only because they share a few outlier
residual dimensions?  Remove the top-k dims (by mean u^2 over all directions) and recompute |cos|.
Also: how similar are generic weight directions (top left singular vectors of every down_proj) - the anisotropy baseline.
    python control_outlier_dims.py <base safetensors> <edited dir> <exp8 code dir>
"""
import glob, os, sys
import numpy as np
base_path, edited, code_dir = sys.argv[1:4]
sys.path.insert(0, code_dir)
import st_numpy as S
P = "model.language_model."
names = [P + f"layers.{l}.mlp.down_proj.weight" for l in range(24)] + [P + "layers.15.self_attn.o_proj.weight"]
files = {(r, os.path.basename(f)[:-4]): f for r in ["p10c", "p50c"] for f in glob.glob(f"{edited}/{r}/*.npy")}
names = sorted(set(names) | {n for _, n in files})
B = {k: S.bf16_to_f32(b).astype(np.float64) for k, (_, b) in S.load(base_path, names).items()}
D = {}
for (r, n), f in files.items():
    U = np.linalg.svd(np.load(f).astype(np.float64) - B[n], full_matrices=False)[0][:, 0]
    D[(r, n)] = U
p50 = [D[k] for k in sorted(D) if k[0] == "p50c"]
p10 = [D[k] for k in sorted(D) if k[0] == "p10c"]
allu = np.stack(p10 + p50)
mass = (allu ** 2).mean(0)
order = np.argsort(-mass)
# generic baseline: top left singular vector of each layer's down_proj (directions the MLPs write most)
gen = np.stack([np.linalg.svd(B[P + f"layers.{l}.mlp.down_proj.weight"], full_matrices=False)[0][:, 0] for l in range(24)])
# second baseline: per-layer "most written residual dims" direction = row norms of W_down (positive vector)
rown = np.stack([np.linalg.norm(B[P + f"layers.{l}.mlp.down_proj.weight"], axis=1) for l in range(24)])
rown /= np.linalg.norm(rown, axis=1, keepdims=True)
def cosm(X, Y, keep):
    X = X[:, keep]; Y = Y[:, keep]
    X = X / np.linalg.norm(X, axis=1, keepdims=True); Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    return np.abs(X @ Y.T)
print("dims ranked by mean u^2:", order[:12].tolist(), " their share:", np.round(mass[order[:12]] / mass.sum(), 3).tolist())
print(f"{'removed':>8} | p50c vs p10c median | p50c u15-u19 | p10c-p10c median | generic(svd1) vs p50c max | null 99.9%")
rng = np.random.default_rng(0)
for k in [0, 5, 10, 20, 50, 100]:
    keep = np.setdiff1d(np.arange(1024), order[:k])
    a = cosm(np.stack(p50), np.stack(p10), keep)
    b = cosm(np.stack(p50), np.stack(p50), keep)[0, 1]
    c = cosm(np.stack(p10), np.stack(p10), keep); c = c[~np.eye(len(p10), dtype=bool)]
    g = cosm(gen, np.stack(p50), keep).max()
    x = rng.standard_normal((20000, len(keep))); y = rng.standard_normal((20000, len(keep)))
    null = np.quantile(np.abs((x * y).sum(1)) / np.linalg.norm(x, axis=1) / np.linalg.norm(y, axis=1), .999)
    print(f"{k:>8} | {np.median(a):19.3f} | {b:12.3f} | {np.median(c):16.3f} | {g:25.3f} | {null:.3f}")
print("cos(p50c u, row-norm profile of its own layer):",
      [round(float(abs(D[('p50c', P + f'layers.{l}.mlp.down_proj.weight')] @ rown[l])), 3) for l in (15, 19)])
# p25c head alignment after removing top-20 dims
Wo = B[P + "layers.15.self_attn.o_proj.weight"]
keep = np.setdiff1d(np.arange(1024), order[:20])
for lab, u in [("p50c u15", p50[0]), ("p50c u19", p50[1])]:
    uu = u[keep] / np.linalg.norm(u[keep]); W = Wo[keep]
    fr = [np.sum((uu @ W[:, h*256:(h+1)*256])**2) / np.sum(W[:, h*256:(h+1)*256]**2) * len(keep) for h in range(8)]
    print(f"head alignment, top-20 dims removed, {lab}: " + " ".join(f"h{h}={f:.2f}" for h, f in enumerate(fr)))
