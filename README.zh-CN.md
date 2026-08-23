# CIFAR-100 持续学习项目

这是一个面向科研入门、可复现的类增量学习项目。模型顺序学习 CIFAR-100 的十个任务，每个任务包含十个新类别；学完每个任务后，程序重新测试所有已经见过的任务，由此直接观察灾难性遗忘。

[English README](README.md) · [英文研究短报告](report/continual_learning_cifar100_report.pdf) · [详细实验记录](RESULTS.md)

## 当前结论

在确定性的 seed-42 方法比较中：

| 方法 | 最终平均准确率 | 最终平均遗忘 |
|---|---:|---:|
| Naive Fine-Tuning | 6.64% | 59.32% |
| Replay, buffer=2,000 | **17.96%** | **54.90%** |
| Online EWC, lambda=10 | 6.37% | 57.46% |

对 Replay buffer 1,000、2,000、5,000 使用 seeds `0 / 1 / 42` 复核后，最终平均准确率分别为 `12.14% ± 1.40%`、`16.85% ± 1.18%`、`26.79% ± 1.56%`。三个 seed 都保持相同的递增顺序。

![三随机种子 Replay buffer 消融](runs/replay-multiseed/buffer-size.svg)

## 项目实现

- CIFAR-100 十任务、单头 class-incremental evaluation；
- 适配 32×32 图像的 ResNet-18；
- Naive、Experience Replay、Online EWC；
- reservoir-sampling replay buffer；
- accuracy matrix、Average Accuracy、Average Forgetting；
- 可断点复用的 buffer-size 与多 seed 消融；
- JSON、CSV、checkpoint 和 SVG 实验产物；
- 确定性 CUDA 与非有限 loss 防护；
- 八项自动测试。

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

下一项方法实验会把 buffer 固定为 2,000，在相同随机种子和训练预算下比较当前均匀回放与一种课程式或重要性采样策略，从“存多少样本”推进到“应回放哪些样本”。DER++ 保留为第二基线；另一个工程方向是将 ReplayBuffer 改为紧凑 `uint8` 原图存储，并在 replay 时动态增强。
