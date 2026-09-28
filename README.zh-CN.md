# CIFAR-100 持续学习项目

这是一个面向科研入门、可复现的类增量学习项目。模型顺序学习 CIFAR-100 的十个任务，每个任务包含十个新类别；学完每个任务后，程序重新测试所有已经见过的任务，由此直接观察灾难性遗忘。

[English README](README.md) · [英文研究短报告](report/continual_learning_cifar100_report.pdf) · [Tee-Zhang 复现报告](report/TEE_ZHANG_2023_REPRODUCTION_REPORT.md) · [详细实验记录](RESULTS.md)

## 当前结论

在确定性的 seed-42 方法比较中：

| 方法 | 最终平均准确率 | 最终平均遗忘 |
|---|---:|---:|
| Naive Fine-Tuning | 6.64% | 59.32% |
| Replay, buffer=2,000 | **17.96%** | **54.90%** |
| Online EWC, lambda=10 | 6.37% | 57.46% |

对 Replay buffer 1,000、2,000、5,000 使用 seeds `0 / 1 / 42` 复核后，最终平均准确率分别为 `12.14% ± 1.40%`、`16.85% ± 1.18%`、`26.79% ± 1.56%`。三个 seed 都保持相同的递增顺序。

![三随机种子 Replay buffer 消融](runs/replay-multiseed/buffer-size.svg)

## 论文复现：Tee and Zhang (2023)

已经完成论文 Table 1 中 divisions `1 / 120 / 300` × seeds `0 / 1 / 2 / 3`
的 12 组 ciFAIR-100 正式实验：

| Division | 复现 Avg Acc | 论文 Avg Acc | 复现 F | 论文 F |
|---:|---:|---:|---:|---:|
| 1 | 39.59% ± 0.42% | 39.9% | 64.69% ± 0.52% | 63.9% |
| 120 | **46.49% ± 0.75%** | 46.6% | **55.50% ± 0.67%** | 55.1% |
| 300 | 46.38% ± 0.55% | 46.6% | 56.70% ± 1.07% | 56.1% |

六项 headline metrics 与论文的最大差距为 0.79 个百分点，并复现了 division 120
优于 division 1、division 300 不再提高准确率的主趋势。

![论文值与四 seed 复现值](report/figures/tee_zhang_2023_paper_comparison.png)

### Equal-budget 对照实验

发布代码在 division 120/300 下每 epoch 多呈现 90/150 个 current examples。`equal-budget`
模式让每个样本每 epoch 恰好出现一次。Windows 与 WSL 的结果不逐位一致（CPU bicubic
resize 的浮点舍入不同，同一 seed 可相差 1.6 个百分点），因此两组都在 WSL 上按 seed
`0–7` 配对重跑（共 32 组）。division 1 在两种模式下序列相同，它的 WSL 结果作为共同参照。

| Division | released-code Avg Acc | equal-budget Avg Acc | 配对差值 (95% CI) | released-code F | equal-budget F | 配对差值 (95% CI) |
|---:|---:|---:|---:|---:|---:|---:|
| 120 | 46.33% ± 0.84% | 46.23% ± 0.42% | −0.10 pp [−0.59, +0.39] | 55.90% ± 1.48% | 55.92% ± 1.47% | +0.01 pp [−1.59, +1.62] |
| 300 | 46.35% ± 0.52% | 46.25% ± 0.60% | −0.10 pp [−0.40, +0.20] | 55.65% ± 1.40% | 55.82% ± 0.88% | +0.17 pp [−1.10, +1.44] |

![Equal-budget 配对对比](report/figures/tee_zhang_2023_equal_budget.png)

重复样本没有可检测的效应，Avg Acc 配对差值被限制在约 ±0.5 个百分点以内。在相同样本预算下，
division 120/300 仍比同 seed 的 WSL division 1 高 6.36 / 6.39 个百分点，`F` 低 9.15 / 9.25 个百分点（8/8 seed 同向），
因此论文的 interleaving 收益来自 interleaving 本身，而不是样本数量差异。分析脚本：
`tools/analyze_tee_zhang_equal_budget.py`。

### 完整 Table 1 曲线（同一平台）

在 WSL 上补齐 divisions `8 / 60` 并重跑 division 1，五个 division 各 seed `0–7`（共 40 组 released-code 实验）。

| Division | 复现 Avg Acc | 论文 Avg Acc | 复现 F | 论文 F |
|---:|---:|---:|---:|---:|
| 1 | 39.86% ± 0.63% | 39.9% | 65.07% ± 1.61% | 63.9% |
| 8 | 40.26% ± 0.62% | 40.7% | 63.85% ± 0.91% | 62.6% |
| 60 | 44.90% ± 0.37% | 44.6% | 57.47% ± 1.05% | 57.4% |
| 120 | 46.33% ± 0.84% | 46.6% | 55.90% ± 1.48% | 55.1% |
| 300 | 46.35% ± 0.52% | 46.6% | 55.65% ± 1.40% | 56.1% |

![完整 Table 1 曲线](report/figures/tee_zhang_2023_wsl_curve.png)

五个 Avg Acc 均值与论文相差都在 0.44 个百分点以内，完整复现了曲线形状。按 seed 配对：division 1→8 提升
+0.40 个百分点（95% CI [+0.08, +0.71]，只有论文 +0.8 的一半，需要 8 个 seed 才能分辨），8→60 提升 +4.65，
60→120 提升 +1.42，120→300 进入平台（+0.02 [−0.52, +0.57]）。`F` 在 division 1/8/120 上比论文高 0.8–1.3
个百分点，是 Windows 复现中也出现过的小而稳定的偏移。分析脚本：`tools/analyze_tee_zhang_curve.py`。

### Replay 采样策略：uniform vs loss 优先级

buffer 固定为 2,000（reservoir 存储，每步回放 64 个），两组只在"抽哪些样本回放"上不同。`loss` 组采用 prioritized replay：
优先级 `(最近回放 loss + 1e-3) ** 0.6`，加权不放回抽样，新样本取当前最大优先级，α 在实验前固定
（[预注册](docs/REPLAY_SAMPLING_PLAN.md)）。两组都在 WSL 上按 seed `0–7` 配对运行。

| 策略 | 最终平均准确率 | 最终平均遗忘 |
|---|---:|---:|
| uniform | 17.14% ± 1.66% | 55.05% ± 2.07% |
| loss | 17.74% ± 1.19% | 54.03% ± 1.79% |
| 配对差值 loss − uniform (95% CI) | +0.59 pp [−0.98, +2.17]，p = 0.40 | −1.01 pp [−2.56, +0.53]，p = 0.16 |

没有检测到收益，效应被限制在约 −1 到 +2 个百分点之间。诊断给出了原因：训练 task 10 时，task 1–8 的平均优先级只有 0.058
（uniform 为 1），模型已经记住了 buffer 中的样本，buffer 上的 loss 不再反映测试集上的遗忘，优先级因此主要把抽样推向最近的旧任务
（task 9 份额 11.3% → 17.1%）。数据：`runs/replay-sampling/`；runner：`run_replay_sampling.py`。

## 项目实现

- CIFAR-100 十任务、单头 class-incremental evaluation；
- 适配 32×32 图像的 ResNet-18；
- Naive、Experience Replay、Online EWC；
- reservoir-sampling replay buffer；
- accuracy matrix、Average Accuracy、Average Forgetting；
- 可断点复用的 buffer-size 与多 seed 消融；
- JSON、CSV、checkpoint 和 SVG 实验产物；
- 确定性 CUDA 与非有限 loss 防护；
- 十三项自动测试，包括论文 class order、interleave sequencing、replay quota 与论文指标公式。

## 快速验证

```powershell
conda run -n ece488_clip python train.py --dataset synthetic --num-classes 6 --classes-per-task 2 --epochs-per-task 1 --method naive --model tiny --output-dir runs/smoke-naive
conda run -n ece488_clip python -m unittest discover -s tests -v
```

三随机种子 Replay 消融：

```powershell
conda run --no-capture-output -n ece488_clip python run_replay_ablation.py --buffer-sizes 1000 2000 5000 --seeds 0 1 42 --output-dir runs/replay-multiseed
```

完整配置、逐任务矩阵、失败实验和结论边界见 [RESULTS.md](RESULTS.md)。

## 后续研究

论文复现已经完成：同一平台上复现了完整的 Table 1 曲线，`equal-budget` 对照实验也支持 interleaving 解释。第一个方法层面的对比（buffer 2,000 下的 uniform 与 loss 优先级回放）没有检测到效应，原因是 buffer 样本被记住后 loss 失去信号。下一步是换一个不会被记忆掩盖的优先级信号，例如对存储样本的新增强视图计算 loss，或统计遗忘事件次数。DER++ 保留为第二基线；另一个工程方向是将 ReplayBuffer 改为紧凑 `uint8` 原图存储，并在 replay 时动态增强。
