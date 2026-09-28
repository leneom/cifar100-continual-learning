# Tee & Zhang (2023) Interleave-Division Reproduction

## Research question

When the current-task dataset `D_t` and the old replay buffer `R_t` contain the
same examples and are exposed for the same number of epochs, does more frequent
interleaving reduce catastrophic forgetting and improve continual accuracy?

The target is Table 1 of:

> Ren Jie Tee and Mengmi Zhang. "Integrating Curricula with Replays: Its
> Effects on Continual Learning." AAAI Summer Symposium Series, 2023.

Official paper: <https://ojs.aaai.org/index.php/AAAI-SS/article/view/27486>

Official code: <https://github.com/ZhangLab-DeepNeuroCogLab/Integrating-Curricula-with-Replays>

## Target result

| Interleave division | 1 | 8 | 60 | 120 | 300 |
|---:|---:|---:|---:|---:|---:|
| Paper forgetfulness `F` (%) | 63.9 | 62.6 | 57.4 | 55.1 | 56.1 |
| Paper average accuracy (%) | 39.9 | 40.7 | 44.6 | 46.6 | 46.6 |

The first bounded reproduction uses divisions `1 / 120 / 300`. It succeeds if
four-seed means reproduce the qualitative claims:

1. division 120 has higher continual average accuracy and lower forgetfulness
   than division 1;
2. division 300 provides little or no accuracy gain over division 120;
3. all protocol differences and failed or divergent runs remain visible.

Exact numerical equality is not required because the released environment,
GPU kernels, and torchvision weights are older than the current environment.

## Locked paper protocol

- Dataset: ciFAIR-100.
- Stream: 20 tasks, 5 new classes per task.
- Class-order seed: 100, fixed across every run.
- Model: ImageNet-pretrained torchvision MobileNetV3-Small.
- Output head: starts with 5 classes and expands by 5 after every task.
- Training split: 450 images per class; validation: final 50 train images per
  class; test: the standard corrected ciFAIR-100 test split.
- Replay memory: 1,200 images, rebalanced equally across all seen classes after
  each task.
- Optimizer: SGD, learning rate 0.001, momentum 0.9, no weight decay.
- Batch size: 32.
- Stopping: retain the lowest current-task validation-loss checkpoint and stop
  after 5 consecutive non-improving epochs.
- Seeds: 4 controlled seeds per condition.

## What an interleave division means

For each task, shuffle the current-task examples and the old replay examples
independently. Split each sequence into `d` chunks and concatenate them as:

```text
current chunk 1, replay chunk 1,
current chunk 2, replay chunk 2,
...
current chunk d, replay chunk d
```

`d=1` therefore presents the entire current task before the entire replay
buffer. Larger `d` switches between new and old examples more frequently.
The constructed order is fixed across all epochs of that task, matching the
released implementation.

## Paper metrics

Let `a_t` be accuracy on all classes seen after task `t`, and let `b_t` be
accuracy on the five classes from task 1 after task `t`.

```text
continual average accuracy = mean_t(a_t)
forgetfulness F           = mean_{t=2..20}((b_1 - b_t) / b_1)
```

The exact `F` formula above was confirmed in the authors' released
`ciFAIR-100/Plots/varydiv.ipynb`: it computes relative percentage decrease and
excludes Task 1's zero-forgetting row. The paper reports these means over tasks
and then averages four runs. These are not the same as this repository's
existing final-average-accuracy and all-task-average-forgetting metrics, so the
reproduction stores both the per-task trajectory and the paper-specific
summary.

## Paper-code discrepancies found during audit

### Image size

The paper appendix says 72x72. `ciFAIR-100/Code/VaryDiv.py` in the released
repository resizes to 74x74. The initial code-following reproduction uses 74
and records it in every result. A 72x72 sensitivity run can be added later.

### Unequal sample budget in released interleaving code

The paper states that every division exposes the same total number of current
and replay samples. The released code first distributes the remainder of the
current-task sequence across chunks and then appends that remainder again.
With 2,250 current-task samples this duplicates:

| Division | Extra current examples per epoch |
|---:|---:|
| 1 | 0 |
| 120 | 90 |
| 300 | 150 |

The reproduction therefore exposes two explicit modes:

- `released-code`: preserves the duplicate-tail behavior to trace the published
  implementation and its Table 1 trend;
- `equal-budget`: uses each current and replay example exactly once per epoch,
  matching the paper's stated experimental control.

This distinction is part of the scientific result, not a cleanup detail.

## Staged execution

1. Unit-test class order, balanced chunking, duplicate-tail counts, replay
   rebalancing, and paper metric definitions using tiny synthetic data.
2. Download and checksum ciFAIR-100 through the official v1.0 loader.
3. Run a short 2-task, 1-epoch smoke test to validate CUDA, labels, expanding
   head, buffer size, checkpoints, and result serialization.
4. Run divisions `1 / 120 / 300` with four seeds in `released-code` mode.
5. Run the matching `equal-budget` sensitivity experiment before interpreting
   the result as evidence for interleaving itself (completed 2026-09-28; see below).

## Completed execution record

### Validation completed

- ciFAIR-100 v1.0 archive MD5:
  `DDC236AB4B12EEB8B20B952614861A33` (matches the official loader).
- Thirteen repository tests pass, including the paper class order, released-code
  duplicate tail, equal-budget sequencing, class-balanced replay quotas, and
  paper metric formulas.
- Two-task smoke tests completed for both `released-code` and `equal-budget`.
  With 160 current examples and division 120, the former presented 200 current
  examples while the latter presented exactly 160, as expected.

### Formal released-code matrix

All 12 planned runs completed for divisions `1 / 120 / 300` and controlled
seeds `0 / 1 / 2 / 3`.

| Division | Reproduction Avg Acc, mean +/- sample std | Paper Avg Acc | Delta | Reproduction F, mean +/- sample std | Paper F | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 39.59% +/- 0.42% | 39.9% | -0.31 pp | 64.69% +/- 0.52% | 63.9% | +0.79 pp |
| 120 | **46.49% +/- 0.75%** | 46.6% | -0.11 pp | **55.50% +/- 0.67%** | 55.1% | +0.40 pp |
| 300 | 46.38% +/- 0.55% | 46.6% | -0.22 pp | 56.70% +/- 1.07% | 56.1% | +0.60 pp |

The bounded reproduction meets its preregistered qualitative success criteria:

1. division 120 raises continual average accuracy by 6.90 points and lowers F
   by 9.19 points relative to division 1;
2. division 300 provides no accuracy gain over division 120 and has slightly
   worse F; and
3. every reported mean is close to the paper, with a maximum absolute gap of
   0.79 percentage points across the six headline metrics.

This supports a successful reproduction of the selected Table 1 conditions
and their ordering. It is not a reconstruction of the complete five-division
curve because divisions 8 and 60 were not run.

### Evidence integrity

- Twelve `results.json` files and twelve 20x20 accuracy matrices are present.
- No NaN or Infinity appears in the saved JSON or CSV evidence.
- The final matrix runner exited with code 0 and empty error logs.
- Every run records the configuration, software and GPU environment, per-task
  trajectories, per-epoch logs, matrix, and checkpoint.
- Two early division-1 elapsed-time values include Windows Modern Standby and
  must not be used for runtime comparison. Their numerical results are complete.

### Metric audit correction

An initial implementation incorrectly treated `F` as an absolute accuracy-point
drop and included Task 1. The raw trajectory was unaffected. The error was found
by auditing the authors' plotting notebook, corrected before launching the
remaining formal runs, and the saved seed-0 summaries were recomputed without
retraining.

### Reproduction command

```powershell
$env:TORCH_HOME = "$PWD\.torch-cache"
conda run --no-capture-output -n ece488_clip python run_tee_zhang_2023_matrix.py `
  --sequence-implementation released-code `
  --divisions 1 120 300 `
  --seeds 0 1 2 3
```

The matrix runner validates matching configurations, skips completed runs, and
updates `summary.json` and `summary.csv` after every successful run.

### Cross-platform determinism check (WSL, 2026-09-27)

The equal-budget sensitivity matrix runs under WSL (Ubuntu 24.04, conda env
`cifar-cl`) with the same torch 2.13.0+cu126, torchvision 0.28.0, Pillow 12.3.0
and NumPy 2.2.6 as the Windows `ece488_clip` environment and the same RTX 4080
Laptop GPU. Before pairing new runs with the Windows matrix, the division-120
seed-0 `released-code` run was repeated under WSL
(`runs/tee-zhang-2023/wsl-platform-check/`).

- Within WSL, two repeated 2-task smoke runs are bit-identical.
- Across platforms, the initial model parameters and the interleave plan hash
  are identical, but the preprocessed input tensors differ at float-rounding
  level (first 32-image batch sum -12013.2610 on WSL vs -12013.2596 on
  Windows). The source is the CPU bicubic tensor resize, whose compiled
  kernels differ between the Linux and Windows builds.
- This perturbation propagates through training: the full division-120 seed-0
  run gives 45.60% Avg Acc and 57.24% F under WSL versus 47.24% and 54.63%
  under Windows, a shift larger than the four-seed sample SD (0.75 pp).

Consequences: determinism holds per platform, not across platforms; paired
comparisons must use runs from one platform; and seed-level differences of
about 1.5 pp should be read as ordinary training noise for this protocol.
The WSL `released-code` baseline for divisions 120/300 is therefore rerun in
the same root, and both arms are extended to seeds 0-7. Division 1 is not
rerun, because its interleave plan is identical under both implementations and
its gap to divisions 120/300 (about 7 pp) is far larger than this noise.

### Equal-budget control result

Both arms completed under WSL for divisions 120/300 and paired seeds 0-7
(`runs/tee-zhang-2023/equal-budget/`, `runs/tee-zhang-2023/wsl-platform-check/released-code/`;
paired statistics in `runs/tee-zhang-2023/equal-budget-comparison/`).

| Division | Paired Avg Acc diff, equal-budget - released (95% CI) | Paired F diff (95% CI) |
|---:|---:|---:|
| 120 | -0.10 pp [-0.59, +0.39] | +0.01 pp [-1.59, +1.62] |
| 300 | -0.10 pp [-0.40, +0.20] | +0.17 pp [-1.10, +1.44] |

The duplicate tail has no detectable effect, and the equal-budget arms still
exceed division 1 by 6.64 / 6.66 pp in accuracy. The division-120 benefit is
therefore attributable to interleaving rather than unequal current-sample
counts. Remaining reproduction gap: divisions 8 and 60.
