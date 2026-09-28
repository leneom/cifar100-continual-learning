# Catastrophic Forgetting in Class-Incremental Learning on CIFAR-100

A reproducible study of how a single-head image classifier forgets old classes when new classes arrive sequentially. The project compares Naive Fine-Tuning, Experience Replay, and Online EWC, then isolates the effect of replay-buffer capacity through controlled ablations and three-seed validation.

[中文说明](README.zh-CN.md) · [Research report](report/continual_learning_cifar100_report.pdf) · [Detailed results](RESULTS.md) · [Learning guide](docs/LEARNING_GUIDE.md)

## Key result

Experience Replay is the strongest tested baseline. In the deterministic seed-42 comparison, Replay with 2,000 stored examples reaches **17.96%** final average accuracy, compared with **6.64%** for Naive Fine-Tuning and **6.37%** for Online EWC.

The replay-capacity trend is consistent across seeds `0`, `1`, and `42`:

| Replay buffer | Final average accuracy | Final average forgetting |
|---:|---:|---:|
| 1,000 | 12.14% ± 1.40% | 60.56% ± 2.47% |
| 2,000 | 16.85% ± 1.18% | 54.36% ± 0.81% |
| 5,000 | **26.79% ± 1.56%** | **42.63% ± 1.51%** |

![Three-seed replay-buffer ablation](runs/replay-multiseed/buffer-size.svg)

All three seeds preserve the ordering `1,000 < 2,000 < 5,000` in final average accuracy. Larger replay memory therefore produces a robust improvement in this configuration, although it does not eliminate forgetting.

## Paper reproduction: Tee and Zhang (2023)

The project now includes a four-seed reproduction of selected ciFAIR-100 conditions from Table 1 of *Integrating Curricula with Replays: Its Effects on Continual Learning*. The released-code divisions `1 / 120 / 300` were each run with seeds `0 / 1 / 2 / 3`.

| Division | Reproduction Avg Acc | Paper Avg Acc | Reproduction F | Paper F |
|---:|---:|---:|---:|---:|
| 1 | 39.59% ± 0.42% | 39.9% | 64.69% ± 0.52% | 63.9% |
| 120 | **46.49% ± 0.75%** | 46.6% | **55.50% ± 0.67%** | 55.1% |
| 300 | 46.38% ± 0.55% | 46.6% | 56.70% ± 1.07% | 56.1% |

![Paper targets and reproduction](report/figures/tee_zhang_2023_paper_comparison.png)

All six reproduced headline metrics are within 0.79 percentage points of the paper, and the main ordering is recovered: division 120 improves accuracy and forgetting over division 1, while division 300 provides no further accuracy gain. The report explicitly documents two paper-code discrepancies: the released code uses 74×74 images although the appendix states 72×72, and its chunking duplicates current-task tail examples for divisions 120 and 300.

### Equal-budget control

An `equal-budget` sensitivity matrix removes the duplicate tail so that every current and replay example appears exactly once per epoch. Windows and WSL runs are not bit-identical (CPU bicubic resize rounding shifts a single seed by up to 1.6 pp), so both arms were rerun on WSL with paired seeds `0–7` (32 runs). Division 1 is identical under both implementations, so its WSL runs serve as the shared reference.

| Division | released-code Avg Acc | equal-budget Avg Acc | Paired diff (95% CI) | released-code F | equal-budget F | Paired diff (95% CI) |
|---:|---:|---:|---:|---:|---:|---:|
| 120 | 46.33% ± 0.84% | 46.23% ± 0.42% | −0.10 pp [−0.59, +0.39] | 55.90% ± 1.48% | 55.92% ± 1.47% | +0.01 pp [−1.59, +1.62] |
| 300 | 46.35% ± 0.52% | 46.25% ± 0.60% | −0.10 pp [−0.40, +0.20] | 55.65% ± 1.40% | 55.82% ± 0.88% | +0.17 pp [−1.10, +1.44] |

![Equal-budget paired comparison](report/figures/tee_zhang_2023_equal_budget.png)

The duplicate tail has no detectable effect: the paired accuracy difference is bounded to roughly ±0.5 pp. Under equal budgets, divisions 120 and 300 still exceed the same-seed WSL division 1 by 6.36 and 6.39 pp and lower F by 9.15 and 9.25 pp (8/8 seeds), so the paper's interleaving gain is attributable to interleaving rather than unequal sample counts. Analysis: `tools/analyze_tee_zhang_equal_budget.py`; paired results in `runs/tee-zhang-2023/equal-budget-comparison/`.

### Full Table 1 curve (same platform)

Divisions `8 / 60` were added and division 1 rerun on WSL, giving all five Table 1 divisions with seeds `0–7` (40 released-code runs).

| Division | Reproduction Avg Acc | Paper Avg Acc | Reproduction F | Paper F |
|---:|---:|---:|---:|---:|
| 1 | 39.86% ± 0.63% | 39.9% | 65.07% ± 1.61% | 63.9% |
| 8 | 40.26% ± 0.62% | 40.7% | 63.85% ± 0.91% | 62.6% |
| 60 | 44.90% ± 0.37% | 44.6% | 57.47% ± 1.05% | 57.4% |
| 120 | 46.33% ± 0.84% | 46.6% | 55.90% ± 1.48% | 55.1% |
| 300 | 46.35% ± 0.52% | 46.6% | 55.65% ± 1.40% | 56.1% |

![Full Table 1 curve](report/figures/tee_zhang_2023_wsl_curve.png)

Every accuracy mean is within 0.44 pp of the paper and the full shape is recovered. Paired over seeds, accuracy rises by +0.40 pp from division 1 to 8 (95% CI [+0.08, +0.71]; half the paper's +0.8 pp and only resolvable with eight seeds), +4.65 pp from 8 to 60, +1.42 pp from 60 to 120, and plateaus from 120 to 300 (+0.02 pp [−0.52, +0.57]). F is 0.8–1.3 pp above the paper at divisions 1, 8 and 120, a small consistent offset also seen in the Windows runs. Analysis: `tools/analyze_tee_zhang_curve.py`; summary in `runs/tee-zhang-2023/wsl-curve/`.


See the [protocol and audit](docs/TEE_ZHANG_2023_REPRODUCTION.md), [editable reproduction report](report/TEE_ZHANG_2023_REPRODUCTION_REPORT.md), and [machine-readable summary](runs/tee-zhang-2023/released-code/summary.csv). The local PDF can be regenerated with `python tools/build_tee_zhang_reproduction_report.py`.

## Replay sampling: uniform vs loss-prioritized

With the buffer fixed at 2,000 examples (reservoir storage, 64 replayed per step), the only difference between arms is which stored examples are drawn. The `loss` arm uses prioritized replay: priority `(last replay loss + 1e-3) ** 0.6`, weighted draws without replacement, new slots at the running maximum priority, with alpha fixed before any run ([pre-registration](docs/REPLAY_SAMPLING_PLAN.md)). Both arms ran on WSL with paired seeds `0–7`.

| Policy | Final average accuracy | Final average forgetting |
|---|---:|---:|
| uniform | 17.14% ± 1.66% | 55.05% ± 2.07% |
| loss | 17.74% ± 1.19% | 54.03% ± 1.79% |
| Paired diff, loss − uniform (95% CI) | +0.59 pp [−0.98, +2.17], p = 0.40 | −1.01 pp [−2.56, +0.53], p = 0.16 |

No benefit is detected; the effect is bounded to roughly −1 to +2 pp. The diagnostics explain why: while training task 10, the mean priority of tasks 1–8 is only 0.058 (uniform: 1.0), so the model has memorized the stored examples and buffer loss no longer tracks test-set forgetting. Prioritization therefore mostly shifts draws toward the most recent old task (task 9 share 11.3% → 17.1%). Data: `runs/replay-sampling/`; runner: `run_replay_sampling.py`.

## Research question

The project studies a simple but consequential question:

> How does replay-buffer capacity affect catastrophic forgetting in single-head class-incremental image classification?

CIFAR-100 is split into ten tasks of ten classes. A ResNet-18 learns the tasks sequentially and is evaluated on every previously observed task after each training stage. The resulting lower-triangular accuracy matrix exposes both retained knowledge and catastrophic forgetting.

## Experimental design

- **Dataset:** CIFAR-100, 50,000 training images and 10,000 test images.
- **Scenario:** ten class-incremental tasks, ten new classes per task.
- **Evaluation:** one shared 100-class output head; no task identity at test time.
- **Model:** ResNet-18 adapted for 32×32 images.
- **Optimization:** ten epochs per task, SGD, learning rate 0.1, momentum 0.9, weight decay `5e-4`, batch size 128.
- **Methods:** Naive Fine-Tuning, reservoir-sampling Experience Replay, and Online diagonal EWC.
- **Fairness:** fixed class order per seed and deterministic CUDA algorithms by default.
- **Statistics:** sample standard deviation across seeds `0`, `1`, and `42` for the key replay capacities.

## What is implemented

- deterministic ten-task training and evaluation;
- TinyConvNet smoke tests and ResNet-18 full experiments;
- Naive, Replay, and Online EWC training paths;
- reservoir-sampling replay memory;
- average-accuracy and average-forgetting metrics;
- accuracy-matrix, JSON, CSV, checkpoint, and SVG artifacts;
- resumable buffer-size and multi-seed ablations;
- guards against non-finite training loss;
- thirteen automated tests for task construction, metrics, Replay, EWC, ablation matching, statistics, plotting, paper class order, interleave sequencing, replay quotas, and paper-specific metrics.

## Reproduce the project

The validated local environment is named `ece488_clip`. Run the fast synthetic smoke test first:

```powershell
conda run -n ece488_clip python train.py --dataset synthetic --num-classes 6 --classes-per-task 2 --epochs-per-task 1 --method naive --model tiny --output-dir runs/smoke-naive
conda run -n ece488_clip python -m unittest discover -s tests -v
```

Run the deterministic CIFAR-100 baselines:

```powershell
conda run --no-capture-output -n ece488_clip python train.py --dataset cifar100 --method naive --epochs-per-task 10 --seed 42 --output-dir runs/cifar100-naive-deterministic
conda run --no-capture-output -n ece488_clip python train.py --dataset cifar100 --method replay --epochs-per-task 10 --buffer-size 2000 --replay-batch-size 64 --seed 42 --output-dir runs/cifar100-replay-2000-deterministic
conda run --no-capture-output -n ece488_clip python train.py --dataset cifar100 --method ewc --epochs-per-task 10 --ewc-lambda 10 --ewc-decay 0.9 --seed 42 --output-dir runs/cifar100-ewc-10-deterministic
```

Run or resume the three-seed replay ablation:

```powershell
conda run --no-capture-output -n ece488_clip python run_replay_ablation.py --buffer-sizes 1000 2000 5000 --seeds 0 1 42 --output-dir runs/replay-multiseed
```

The ablation runner searches existing result files and reuses only runs whose full comparison configuration matches the requested experiment.

## Metrics

Let `A[k, j]` be the accuracy on task `j` after training task `k`.

- **Average Accuracy:** the mean accuracy over all tasks learned by stage `k`.
- **Average Forgetting:** for each previous task, its best historical accuracy minus its current accuracy, averaged over previous tasks.

Both are computed directly from the saved accuracy matrix rather than reconstructed from headline scores.

## Repository structure

```text
continual_learning/       Core data, model, buffer, EWC, metrics, and experiment code
tests/                    Unit tests for the research pipeline
runs/                     Result JSON, accuracy matrices, plots, and local checkpoints
docs/LEARNING_GUIDE.md    Conceptual guide and experiment workflow
report/                   Professor-facing short research report
train.py                  Main experiment entry point
run_replay_ablation.py    Resumable capacity and multi-seed study
run_tee_zhang_2023_matrix.py  Resumable paper-reproduction matrix
plot_results.py           Method-comparison visualization
RESULTS.md                Detailed observations, failures, and limitations
```

## Interpretation and limitations

- Replay substantially improves retention, but early tasks still lose considerable accuracy.
- Online EWC is stable at `lambda=10` but does not improve over Naive in the tested single-head setup. This is a configuration-specific finding, not a general claim that EWC is ineffective.
- An earlier EWC run with `lambda=1000` diverged and produced non-finite loss; it was rejected rather than included as a result.
- The method comparison is still seed-42 only. Three-seed statistics currently cover replay capacities 1,000, 2,000, and 5,000.
- The current buffer stores augmented `float32` tensors. Storing compact source images and applying fresh replay-time augmentation is an open experimental direction.

## Next research step

The paper reproduction is now complete: the full Table 1 curve is recovered on one platform, and the equal-budget control supports the interleaving interpretation. The first method-level comparison (uniform vs loss-prioritized replay at buffer 2,000) found no detectable effect because buffer loss collapses once the stored examples are memorized. The next step is a priority signal that memorization cannot mask, such as loss on fresh augmentations of stored examples or counts of forgetting events. DER++ remains a useful secondary baseline, while a separate systems experiment will test compact `uint8` storage with dynamic replay-time augmentation.

This repository is an independent learning-and-research project intended to demonstrate a complete experimental workflow: formulate a question, implement baselines, control comparisons, retain failed runs, quantify uncertainty, and state the limits of the evidence.
