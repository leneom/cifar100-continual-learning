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

All six reproduced headline metrics are within 0.79 percentage points of the paper, and the main ordering is recovered: division 120 improves accuracy and forgetting over division 1, while division 300 provides no further accuracy gain. The report explicitly documents two paper-code discrepancies: the released code uses 74×74 images although the appendix states 72×72, and its chunking duplicates current-task tail examples for divisions 120 and 300. The next sensitivity experiment therefore uses the repository's `equal-budget` implementation.

See the [protocol and audit](docs/TEE_ZHANG_2023_REPRODUCTION.md), [editable reproduction report](report/TEE_ZHANG_2023_REPRODUCTION_REPORT.md), and [machine-readable summary](runs/tee-zhang-2023/released-code/summary.csv). The local PDF can be regenerated with `python tools/build_tee_zhang_reproduction_report.py`.

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

The immediate paper-reproduction extension reruns divisions `1 / 120 / 300` with paired seeds under the `equal-budget` sequence implementation. This removes the released code's duplicate-tail exposure and tests whether the division-120 advantage is attributable to interleaving rather than unequal sample counts. The broader method-level extension then fixes the buffer at 2,000 examples and compares uniform replay with one curriculum- or importance-aware sampling policy. DER++ remains a useful secondary baseline, while a separate systems experiment will test compact `uint8` storage with dynamic replay-time augmentation.

This repository is an independent learning-and-research project intended to demonstrate a complete experimental workflow: formulate a question, implement baselines, control comparisons, retain failed runs, quantify uncertainty, and state the limits of the evidence.
