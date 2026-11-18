# Exp 8 of "Sparks of Weight-Space Programming": the math gain is loop removal

This repository takes a closer look at Experiment 8 of *Sparks of Weight-Space Programming*, using the authors' released logs ([RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs)). In that experiment, a model edits the weights of Qwen3.5-0.8B and improves its math accuracy. This repository holds the code, the outputs and the generations for four small analyses of that result.

**Main finding.** Almost all of the gain comes from removing the repetition loops that Qwen3.5-0.8B falls into in thinking mode. The evidence:

- **Turning thinking off gets most of the way.** It recovers 81% of the gain of the best edit (p50c) with no weight change.
- **In non-thinking mode the edit does nothing.**
- **The direction can be found without an LLM.** A plain difference of means gives a "repetition direction" with cosine about 0.7 to the edit's direction. Editing along it recovers 84% of p50c's gain.

| Model / edit | Protocol | Accuracy (400 problems) | Cut at token limit |
|---|---|---|---|
| base | thinking (Exp 8's protocol) | 11.8% | 89.5% |
| base | non-thinking (model card default) | 48.5% | 22.0% |
| Astra p50c | thinking | 57.0% | 23.3% |
| Astra p50c | non-thinking | 49.0% | 20.3% |
| random direction, alpha = 6 | thinking | 11.0% | 88.8% |
| hand-derived direction, alpha = 6 | thinking | 49.8% | 37.3% |

The test set is 200 GSM8K-test and 200 MATH-500 problems (seed 0), with one sample per problem, graded with the release's `mathqa.py`. Paired standard errors are about 2.5 points.

## The four analyses

| Folder | Question | Runs on |
|---|---|---|
| [`item1_eval_analysis/`](item1_eval_analysis) | In the released outputs, which answers did the edits gain? (All of them had hit the token limit in the base model, mostly inside a loop.) | laptop, released logs only |
| [`item2_weight_analysis/`](item2_weight_analysis) | Which directions do the three edits act on, and do independent runs agree? (One residual direction per layer; cosine about 0.7 across runs; robust to removing outlier dimensions.) | laptop, weights only |
| [`colab/`](colab), item 3 | Do decoding changes (non-thinking mode, temperature, repetition penalty) give the same gain? | 1 GPU (A100) |
| [`colab/`](colab), item 4 | Can the direction be derived by hand (difference of means over in-loop and pre-loop activations), and does editing along it work? | 1 GPU (A100) |

[`results/`](results) holds the outputs of items 3 and 4:
- summary tables and figures;
- the full generations (`generations/*.jsonl.gz`);
- per-problem scores (`*.scored.csv`);
- the hand-derived directions (`item4/hand_directions.npz`);
- the run logs.

## Reproducing

### Items 1 and 2 (CPU, about 2 GB of disk)

```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/RikiyaT/lm_understand-logs.git
huggingface-cli download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 \
    --local-dir models/Qwen3.5-0.8B
bash scripts/run_local.sh
```

`scripts/run_local.sh` does three things:
- rebuilds the three evaluated edits bit-exactly with the release's `apply_edit.py`, which hash-checks every tensor;
- runs item 1 on `exp8_math/eval`;
- runs item 2 on the rebuilt tensors.

### Items 3 and 4 (Colab A100)

Both items can be run either from a notebook or from the command line.

**Notebook.** Open `colab/item3_decoding_baselines.ipynb` or `colab/item4_hand_direction.ipynb` on an A100 runtime and run all cells. Each notebook writes the helper `wsp_colab.py` itself, installs vLLM, clones the logs, downloads the model at the pinned revision and resumes from cached outputs.

**Command line.** On a Colab VM (for example one started with the colab CLI), upload `colab/wsp_colab.py` and `colab/run_item3.py` (or `run_item4.py`) to `/content`, then:

```bash
pip install -q -U vllm "math-verify[antlr4_13_2]" datasets
nohup python -u /content/run_item3.py > /content/wsp/item3.log 2>&1 &
```

The runs used vLLM 0.34.0, transformers 5.22.1 and torch 2.14.0 (CUDA 13.0). Item 3 takes about 3 hours on an A100 and item 4 about 2.5 hours. Together they used about 37 Colab compute units.

To rescore the released generations without a GPU:
1. Decompress them into a work folder (`gunzip -k results/item3/generations/*.gz`).
2. Call `wsp_colab.score_file(path, exp8_dir)`. It needs `math-verify` and the logs' `exp8_math/code`.

## Notes and limitations

- **Small sample.** Each condition has one sample on 400 problems from two benchmarks. The harder and non-math sets of Exp 8 were not rerun.
- **Only the direction is independent.** In item 4 the direction is derived by hand, but the edit's layers (15, 19), its 64 neurons and alpha = 6 are copied from p50c.
- **One control seed.** The random-direction control uses a single random direction.
- **Practice data only.** Directions are estimated only on the release's practice problems (GSM8K-train and MATH-train), never on the test set.

## Licenses

The code in this repository is under the MIT License (see [`LICENSE`](LICENSE)). Outputs and generations are under CC BY 4.0. The released logs and `apply_edit.py`/`mathqa.py` belong to the authors of *Sparks of Weight-Space Programming* (CC BY 4.0 / MIT). They are downloaded, not copied here. Problem texts from GSM8K and MATH keep their own licenses (MIT).
