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
优于 division 1、division 300 不再提高准确率的主趋势。发布代码存在 duplicate-tail
sample-count confound，所以下一步是运行 paired-seed `equal-budget` sensitivity matrix。

![论文值与四 seed 复现值](report/figures/tee_zhang_2023_paper_comparison.png)

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

当前优先运行 divisions `1 / 120 / 300` 的 `equal-budget` paired-seed sensitivity matrix，移除发布代码中的 duplicate-tail 样本数量混淆。之后再把 buffer 固定为 2,000，在相同随机种子和训练预算下比较均匀回放与一种课程式或重要性采样策略，从“存多少样本”推进到“应回放哪些样本”。DER++ 保留为第二基线；另一个工程方向是将 ReplayBuffer 改为紧凑 `uint8` 原图存储，并在 replay 时动态增强。
