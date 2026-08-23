# 学习指南与实验记录

## 研究问题

主问题：**在 CIFAR-100 类增量学习中，有限 replay memory 能在多大程度上缓解灾难性遗忘？**

第一组可复现实验只改变 buffer size：

```text
0, 200, 500, 1000, 2000, 5000
```

控制不变的变量包括类别顺序、模型初始化、训练轮数、优化器和数据增强。为了避免把某一个随机划分的偶然性当成结论，正式结果应使用至少三个随机种子。

## Naive Fine-Tuning

学习任务 `t` 时，只最小化当前数据的交叉熵：

```text
L = CE(f(x_t), y_t)
```

同一组参数既承载旧知识，也被新任务梯度更新，因此旧任务决策边界容易被破坏。

## Experience Replay

维护容量固定的旧样本集合 `M`。学习当前 batch 时，同时采样 replay batch：

```text
L = CE(f(concat(x_current, x_memory)), concat(y_current, y_memory))
```

本项目使用 reservoir sampling 更新 buffer。流中第 `n` 个样本以 `capacity / n` 的概率被保留，因此无需预先知道总数据量，每个历史样本也具有相同的入选机会。

## Elastic Weight Consolidation (EWC)

EWC 先用 Fisher 信息的对角近似估计参数对旧知识的重要性，再惩罚重要参数偏离旧值：

```text
L = CE(f(x_t), y_t) + lambda / 2 * sum_i F_i * (theta_i - theta_i_old)^2
```

`lambda` 太小几乎不起作用，太大则会阻碍模型学习新类，甚至在较大学习率下造成数值不稳定。因此 EWC 不只需要和 baseline 比较，也应该对 `lambda` 做小规模消融。本项目当前稳定的起点是 `lambda=10, decay=0.9`；`lambda=1000, decay=1.0` 在第 5 个任务发生了 loss 爆炸，不能作为有效对照。

## 指标

设 `A[k, j]` 是学完任务 `k` 后在任务 `j` 上的准确率。

```text
AverageAccuracy(k) = mean_j<=k A[k, j]
Forgetting_j(k) = max_l<k A[l, j] - A[k, j]
AverageForgetting(k) = mean_j<k Forgetting_j(k)
```

最后一个新任务尚未经历后续学习，因此计算 forgetting 时不把它计入旧任务集合。

## 每次实验记录模板

```text
日期：
Git/代码版本：
方法与超参数：
随机种子：
设备与耗时：
最终平均准确率：
最终平均遗忘：
观察到的现象：
可能原因：
下一次只改变哪个变量：
```

## 最容易踩的坑

- 每个任务单独使用 10 类输出，会把实验偷偷变成更容易的 task-incremental learning。
- 只测试最后一个任务，看不到旧知识是否丢失。
- Replay 在当前任务训练前就加入当前任务样本，会造成不一致的数据使用方式。
- 不固定类别顺序和随机种子，方法之间就不是公平对比。
- 只跑一个 seed，不足以支撑稳定结论。
