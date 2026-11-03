# Adaptation, Visual Grounding and Test-Time Compute in Small Language Models

Three ways to adapt a small pretrained language model, each with its own evaluation:
- **Fine-tuning.** Qwen3-0.6B is fine-tuned on 40,000 MetaMathQA examples with Full FT, Full FT + L2-SP, LoRA (rank 4 and 16) and OPLoRA (rank 16, $k=128$). The code reports math transfer and retention benchmarks, weight-space and activation diagnostics, and a blind qualitative annotation.
- **A modular vision-language model.** SigLIP features feed an MLP connector into a frozen Qwen3-0.6B. It is trained in three stages and tested with complementary image pairs and image controls.
- **Test-time compute.** Repeated sampling, answer selection (self-consistency, rank-weighted consensus with self-certainty or VisConf), adaptive stopping and a live parallel-vs-sequential comparison are run on supplied Qwen2.5-3B rollouts.

| Setting | Result |
|---|---|
| GSM8K, Full FT / L2-SP / LoRA-16 / OPLoRA-16 | 63.4 / 63.7 / 61.6 / 60.8 (baseline 43.4) |
| HellaSwag normalized accuracy, same models | 46.1 / 46.4 / 47.5 / 47.4 (baseline 47.3) |
| VLM correct-image advantage over the partner image | +46.9 pp on template pairs, +16.2 pp on natural VQAv2 pairs |
| MATH held-out accuracy at 32 rollouts, self-consistency / self-certainty + RWC | 73.8 / 75.5 (oracle coverage 91.4) |
| Agreement-based stopping on MATH | 73.6 at 10,139 tokens per question against 73.8 at 21,347 |

GSM8K is zero-shot flexible-extract exact match from lm-evaluation-harness. The VLM numbers are local exact-match and VQA soft accuracy on 64 and 96 test pairs. The MATH numbers are on 800 held-out questions, averaged over ten rollout orders.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q
```

The evaluation, VLM and live test-time runs used Python 3.13.15 with torch 2.11.0+cu128, transformers 5.22.0, peft 0.21.2, datasets 5.1.0, lm_eval 0.4.13 and math-verify 0.9.0, as recorded in `results/task1/eval_env.json`, `results/task2/env.json` and `results/task3/live_env.json`. The other pins are in `requirements.txt`. Every training and evaluation script takes `--mode {smoke,full}`. `smoke` runs a few steps on a small slice on CPU, and `full` is the real run. The seed is 6304 everywhere. Model and dataset revisions are pinned in `data/manifest.json`.

Scripts that save checkpoints take `--store`, either a local folder or `hub:<user>/<repo>` for a private Hugging Face repository (it needs an `HF_TOKEN` with write access).

## Compute

All GPU work ran on NVIDIA Tesla T4 GPUs.
- **Fine-tuning:** the five Task 1 runs and the two evaluation jobs ran as Kaggle kernels (`scripts/kaggle_task1.sh`, `scripts/kaggle_task1_eval.sh`), with the LoRA-4 and LoRA-16 runs on a Colab T4 (`scripts/s5_task1_adapters.sh`).
- **VLM and live inference:** the VLM training and evaluation, and the live test-time experiment, ran as Kaggle kernels (`scripts/kaggle_task2.py`, `scripts/kaggle_task3_live.py`).
- **Offline analysis:** the offline test-time analysis runs on CPU.
- **Checkpoints:** final weights, adapters and resume snapshots are written to the store given by `--store`.

```bash
bash scripts/kaggle_task1.sh full_ft l2sp         # one Kaggle kernel per method, two at a time
bash scripts/kaggle_task1_eval.sh a b             # a: baseline and full-model conditions, b: adapters and diagnostics
```

## Reproducing the results

Run from the repo root in this order. Each command writes its outputs under `results/`.

### Fine-tuning
```bash
python -m task1.setup_inputs                      # model facts, qualitative prompts, frozen activation-probe set
python -m task1.build_data --mode full            # first 40,000 eligible MetaMathQA examples, GSM8K overlap guard
for m in full_ft l2sp lora4 lora16 oplora16; do
  python -m task1.train --method $m --mode full --store hub:<user>/<repo>
done
for c in baseline full_ft@625 full_ft l2sp lora4 lora16 oplora16; do
  python -m task1.evaluate --condition $c --mode full --store hub:<user>/<repo>   # adapters are merged and checked first
done
python -m task1.diagnostics --store hub:<user>/<repo>   # rho_k, eta, R(z) on down_proj, displacement
python -m task1.oplora_leak --store hub:<user>/<repo>   # OPLoRA residual overlap against a float64 SVD
python -m task1.boolq_bias                              # BoolQ yes/no prediction bias
python -m task1.select_excerpts                         # summary of task1/annotations.csv and rule-selected excerpts
```

### Vision-language model
Place the COCO 2014 caption and instance annotations and the VQAv2 questions, annotations and complementary pairs in `data/cache/task2_raw/`.
```bash
python -m task2.build_data --mode full            # caption, template and VQAv2 splits, image-disjoint dev and test
python -m task2.cache_features --mode full        # SigLIP features for every image, computed once
python -m task2.train --mode full --store hub:<user>/<repo>        # three stages with development evaluation
python -m task2.final_eval --mode full --store hub:<user>/<repo>   # four image conditions, pair bootstrap
```

### Test-time compute
Download the supplied rollouts into `data/cache/task3_raw/` and MathVista's `testmini.json` into `data/cache/mathvista_meta/`.
```bash
gdown --folder https://drive.google.com/drive/folders/1utX-Gfi9vvaoi1mi0pzNpDGL1o1fAh23 -O data/cache/task3_raw
for d in math mathvista; do
  python -m task3.load --dataset $d               # answers, groups, seeded dev / held-out split and rollout orders
  python -m task3.offline --dataset $d            # selection rules, coverage, failure decomposition, signals, bootstrap
  python -m task3.adaptive --dataset $d --phase dev
done
# commit task3/configs/adaptive_thresholds.json, then:
for d in math mathvista; do python -m task3.adaptive --dataset $d --phase heldout; done
python -m task3.live --mode full --store hub:<user>/<repo>   # single, parallel and sequential strategies on 50 MATH questions
python -m task3.select_cases
```

### Report tables and figures
```bash
for t in task1 task2 task3; do python -m $t.make_figures; done
```

## Attribution

- **Models:** Qwen3-0.6B, Qwen2.5-3B-Instruct and Qwen2.5-VL-3B-Instruct (Qwen license), and SigLIP-base-patch16-224 (Apache-2.0).
- **Data:** MetaMathQA, GSM8K, HellaSwag, BoolQ, ARC-Easy, COCO 2014, VQAv2 and MathVista. The qualitative prompts are `first_plot_questions.yaml` from https://github.com/emergent-misalignment/emergent-misalignment, and the test-time rollouts are the ATML PA3 data release.
- **Libraries:**
  - lm-evaluation-harness (https://github.com/EleutherAI/lm-evaluation-harness) for all Task 1 benchmarks;
  - math-verify (https://github.com/huggingface/Math-Verify) for MATH answer equivalence;
  - PEFT for the reference LoRA used in the equivalence test.
- **Ported code:** `task2/vqa_norm.py` ports `processPunctuation` and `processDigitArticle` from the official VQA evaluation code (https://github.com/GT-Vision-Lab/VQA, BSD-2-Clause).

Apart from that port, no code was copied from public implementations. Code was written with LLM assistance.

## References

- Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
- Li, Grandvalet and Davoine. Explicit Inductive Bias for Transfer Learning with Convolutional Networks. ICML 2018.
- Xiong and Xie. OPLoRA: Orthogonal Projection LoRA Prevents Catastrophic Forgetting during Parameter-Efficient Fine-Tuning. AAAI 2026.
- Biderman et al. LoRA Learns Less and Forgets Less. TMLR 2024.
- Liu et al. Visual Instruction Tuning. NeurIPS 2023.
- Goyal et al. Making the V in VQA Matter. CVPR 2017.
- Zhai et al. Sigmoid Loss for Language Image Pre-Training. ICCV 2023.
- Wang et al. Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR 2023.
- Kang et al. Scalable Best-of-N Selection for Large Language Models via Self-Certainty. 2025.
- Usman et al. VisConf: Quality-Aware Consensus for Best-of-N in Vision-Language Model Reasoning. 2026.
- Madaan et al. Self-Refine: Iterative Refinement with Self-Feedback. NeurIPS 2023.
- Aggarwal et al. Let's Sample Step by Step: Adaptive-Consistency for Efficient Reasoning and Coding with LLMs. EMNLP 2023.
