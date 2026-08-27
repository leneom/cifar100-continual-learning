# Reproducing Interleave-Division Replay on ciFAIR-100

**A four-seed audit of Table 1 in Tee and Zhang (2023)**

Independent reproduction note prepared for research discussion

August 2026

## Executive summary

This study reproduces the ciFAIR-100 interleave-division experiment from Ren
Jie Tee and Mengmi Zhang, *Integrating Curricula with Replays: Its Effects on
Continual Learning*. The bounded matrix covers divisions `1 / 120 / 300` with
controlled seeds `0 / 1 / 2 / 3`, using the ordering behavior in the authors'
released code.

All 12 planned runs completed. The six headline metrics are within 0.79
percentage points of the paper, and the reported qualitative ordering is
recovered:

1. division 120 improves continual average accuracy and reduces forgetfulness
   relative to division 1;
2. division 300 provides no accuracy gain over division 120; and
3. the numerical deviations are small and consistent across the three tested
   conditions.

| Division | Reproduction Avg Acc, mean +/- sample std | Paper Avg Acc | Delta | Reproduction F, mean +/- sample std | Paper F | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 39.59% +/- 0.42% | 39.9% | -0.31 pp | 64.69% +/- 0.52% | 63.9% | +0.79 pp |
| 120 | **46.49% +/- 0.75%** | 46.6% | -0.11 pp | **55.50% +/- 0.67%** | 55.1% | +0.40 pp |
| 300 | 46.38% +/- 0.55% | 46.6% | -0.22 pp | 56.70% +/- 1.07% | 56.1% | +0.60 pp |

For accuracy, higher is better. For `F`, lower is better. Delta is reproduction
minus paper, measured in percentage points.

## Experimental protocol

| Component | Locked configuration |
|---|---|
| Dataset and stream | ciFAIR-100; 20 tasks; 5 new classes per task; class-order seed 100 |
| Model | ImageNet-pretrained torchvision MobileNetV3-Small; expanding five-class head |
| Replay | 1,200 images, rebalanced equally across all seen classes |
| Optimization | SGD; LR 0.001; momentum 0.9; batch size 32; patience 5 |
| Comparison | Released-code divisions `1 / 120 / 300`; seeds `0 / 1 / 2 / 3` |
| Execution | Deterministic CUDA; AMP; image size 74 |
| Environment | Python 3.10.20; PyTorch 2.13.0+cu126; torchvision 0.28.0+cu126; RTX 4080 Laptop GPU |

Let `a_t` be accuracy on all classes seen after task `t`, and let `b_t` be
accuracy on the five Task-1 classes after task `t`.

```text
continual average accuracy = mean_{t=1..20}(a_t)
forgetfulness F           = mean_{t=2..20}((b_1 - b_t) / b_1)
```

The Task-1 row is excluded from `F`, following the authors' released plotting
notebook.

## Interpretation

Moving from division 1 to division 120 raises continual average accuracy by
6.90 points and lowers `F` by 9.19 points. Moving further to division 300 does
not increase average accuracy and increases `F` by 1.21 points relative to
division 120. The main result is therefore a benefit from substantial
interleaving, followed by a plateau rather than monotonic improvement.

All reproduction accuracy means are slightly lower than the paper, while all
reproduction `F` means are slightly higher. The largest gap is 0.79 points.
This small consistent shift is compatible with differences in current CUDA
kernels, torchvision pretrained weights, and the original software
environment; it does not change the experimental ordering.

## Audit findings and limitations

### Released-code sample-budget confound

The released implementation duplicates the tail of the current-task sequence
when it constructs chunks. With 2,250 current examples, this exposes 90 extra
current examples per epoch at division 120 and 150 extra at division 300.
This report intentionally preserves that behavior to trace the published code
and Table 1 trend. It therefore cannot yet attribute the gain solely to
interleaving frequency.

### Paper-code image-size discrepancy

The paper appendix states 72x72, while the released `VaryDiv.py` uses 74x74.
This reproduction follows the released code and records image size 74 in every
result.

### Bounded division set

The experiment tests three of the five reported divisions. Divisions 8 and 60
remain untested. The selected conditions test the main qualitative claims but
do not reconstruct the full curve.

### Runtime caveat

Two early division-1 runs contain Windows Modern Standby intervals. Their
metrics and saved matrices are complete, but their elapsed-time values are not
valid for speed comparison.

### Required sensitivity experiment

The next decisive experiment is the matching `equal-budget` matrix. It removes
duplicate-tail exposure while retaining the interleave schedule. If the
division-120 advantage remains, the evidence for interleaving frequency will
be substantially stronger.

## Evidence integrity

- 12/12 planned result cells completed.
- Every run stores a 20x20 accuracy matrix.
- No NaN or Infinity occurs in the JSON or CSV evidence.
- The final controller exited with code 0 and empty error logs.
- Every result records configuration, environment, task trajectory, epoch log,
  matrix, and checkpoint.
- Thirteen repository tests cover class order, sequencing, replay quotas, and
  paper metric definitions.

## Reproduction command

```powershell
$env:TORCH_HOME = "$PWD\.torch-cache"
conda run --no-capture-output -n ece488_clip python run_tee_zhang_2023_matrix.py `
  --sequence-implementation released-code `
  --divisions 1 120 300 `
  --seeds 0 1 2 3
```

The runner validates matching configurations, skips completed cells, and
updates `summary.json` and `summary.csv` after every successful run.

## Reference

R. J. Tee and M. Zhang. *Integrating Curricula with Replays: Its Effects on
Continual Learning.* AAAI Summer Symposium Series, 2023.

- Paper: <https://ojs.aaai.org/index.php/AAAI-SS/article/view/27486>
- Code: <https://github.com/ZhangLab-DeepNeuroCogLab/Integrating-Curricula-with-Replays>
