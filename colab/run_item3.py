"""Item 3 driver for a Colab VM started with the colab CLI (same steps as item3_decoding_baselines.ipynb).
Run in the background on the VM:  nohup python /content/run_item3.py > /content/wsp/item3.log 2>&1 &
Resumable: rerunning skips finished conditions and finished problems.
"""
import os
import sys

sys.path.insert(0, "/content")
import wsp_colab as W  # noqa: E402

WORK = "/content/wsp"
os.makedirs(f"{WORK}/item3", exist_ok=True)
N_GSM, N_MATH = 200, 200
CONDITIONS = [
    ("base | think35", "base", "think35"),
    ("p50c | think35", "p50c", "think35"),
    ("base | nothink35", "base", "nothink35"),
    ("p50c | nothink35", "p50c", "nothink35"),
    ("base | think_t1", "base", "think_t1"),
    ("base | think35_rep", "base", "think35_rep"),
]


def out_path(name):
    return f"{WORK}/item3/{name.replace(' | ', '__')}.jsonl"


exp8 = W.get_logs()
base = W.get_base()
CKPT = {"base": base, "p50c": W.official_edit_ckpt("p50c", exp8, base, "/content/ckpts/p50c")}
problems_path = f"{WORK}/problems_test.jsonl"
probs = W.load_test(problems_path, N_GSM, N_MATH)
print(len(probs), "problems", flush=True)

# smoke test: 8 problems, non-thinking
smoke = f"{WORK}/smoke_problems.jsonl"
W._write_jsonl(smoke, probs[:4] + probs[-4:])
W.generate(base, smoke, "nothink35", f"{WORK}/smoke_out.jsonl")
print(W.score_file(f"{WORK}/smoke_out.jsonl", exp8, cache=False)[["id", "correct", "pred", "cut", "n_tokens"]], flush=True)
print("SMOKE OK", flush=True)

for name, ck, proto in CONDITIONS:
    print("=" * 20, name, flush=True)
    W.generate(CKPT[ck], problems_path, proto, out_path(name))
    d = W.score_file(out_path(name), exp8)
    print(f"RESULT {name}: acc={d.correct.mean():.3f} cut={d.cut.mean():.3f} tokens={d.n_tokens.mean():.0f}", flush=True)

S = W.summarize({n: W.score_file(out_path(n), exp8) for n, _, _ in CONDITIONS}, "base | think35")
S.to_csv(f"{WORK}/item3/summary.csv")
print(S.to_string(), flush=True)
print("ALL DONE", flush=True)
