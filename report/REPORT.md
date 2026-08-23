# A Reproducible Study of Catastrophic Forgetting in Class-Incremental Learning on CIFAR-100

**Baselines, Replay-Buffer Ablation, and Multi-Seed Validation**  
Independent research project prepared for research discussions at NTU CCDS  
August 2026

## Abstract

Continual-learning systems must acquire new knowledge without erasing what they learned earlier. This project studies catastrophic forgetting in a strict single-head class-incremental setting: CIFAR-100 is divided into ten sequential tasks, while a ResNet-18 always predicts over all 100 classes without receiving task identity at test time. I implemented and compared Naive Fine-Tuning, Experience Replay, and Online Elastic Weight Consolidation (EWC), retaining complete accuracy matrices and using deterministic CUDA execution for fair comparisons. At seed 42, Replay with a 2,000-example memory reached 17.96% final average accuracy, compared with 6.64% for Naive and 6.37% for Online EWC. A controlled replay-capacity ablation showed continued gains up to 5,000 examples. Across seeds 0, 1, and 42, mean final accuracy increased from 12.14% at buffer 1,000 to 26.79% at buffer 5,000, while mean forgetting fell from 60.56% to 42.63%. The results support replay capacity as a robust determinant of performance in this configuration, while also exposing persistent forgetting and an unresolved accuracy-memory trade-off.

## 1. Research question and contribution

The central question is: **How does replay-buffer capacity affect catastrophic forgetting in single-head class-incremental image classification?**

The project contributes a compact, reproducible experimental pipeline rather than a new continual-learning algorithm. Its main contributions are:

1. a deterministic ten-task CIFAR-100 benchmark with full post-task evaluation;
2. comparable implementations of Naive Fine-Tuning, reservoir-sampling Replay, and Online EWC;
3. a controlled buffer-size ablation and three-seed uncertainty estimates; and
4. explicit retention of failed or invalid experiments instead of selective reporting.

## 2. Experimental setup

| Component | Configuration |
|---|---|
| Dataset and stream | CIFAR-100; ten tasks; ten randomly ordered classes per task |
| Evaluation | Single shared 100-class head; no task identity at test time |
| Backbone | ResNet-18 adapted for 32x32 images |
| Optimization | Ten epochs per task; SGD; LR 0.1; momentum 0.9; weight decay 5e-4 |
| Batch policy | Current-task batch 128; replay batch 64 |
| Replay | Fixed-capacity reservoir sampling |
| EWC | Online diagonal Fisher; lambda 10; decay 0.9 |
| Reproducibility | Fixed class order per seed; deterministic CUDA algorithms |

Let A[k,j] denote accuracy on task j after training task k. Average Accuracy is the mean of A[k,j] over learned tasks. Average Forgetting is the mean, over previous tasks, of the best historical accuracy minus the current accuracy. Every run stores the accuracy matrix, configuration, training history, final checkpoint, and summary metrics.

## 3. Results

### 3.1 Baseline comparison

| Method, seed 42 | Final average accuracy | Final average forgetting | Training time |
|---|---:|---:|---:|
| Naive Fine-Tuning | 6.64% | 59.32% | 11.3 min |
| Replay, buffer 2,000 | **17.96%** | **54.90%** | 14.0 min |
| Online EWC, lambda 10 | 6.37% | 57.46% | 12.1 min |

Naive Fine-Tuning exhibits near-complete forgetting: Task 1 falls from 54.4% immediately after it is learned to 0% after Task 2. Replay retains 40.9% on Task 1 after Task 2 and finishes at roughly 2.7 times the final average accuracy of Naive. Online EWC is numerically stable at lambda 10 but provides no meaningful improvement in this configuration. This should be interpreted as a setting-specific result, not as evidence that EWC is generally ineffective.

### 3.2 Replay capacity and multi-seed validation

| Buffer | Final average accuracy, mean ± sample std | Final average forgetting, mean ± sample std |
|---:|---:|---:|
| 1,000 | 12.14% ± 1.40% | 60.56% ± 2.47% |
| 2,000 | 16.85% ± 1.18% | 54.36% ± 0.81% |
| 5,000 | **26.79% ± 1.56%** | **42.63% ± 1.51%** |

The final accuracies for buffers 1,000 / 2,000 / 5,000 are 11.39 / 15.62 / 25.05% for seed 0, 13.75 / 16.98 / 28.07% for seed 1, and 11.28 / 17.96 / 27.26% for seed 42. Every seed therefore preserves the same capacity ordering. The result is stronger than a single best run, but three seeds remain insufficient for precise population-level claims or formal significance testing.

## 4. Failure analysis and limitations

The first nominal method comparison was rejected because GPU nondeterminism caused Naive and Replay to diverge on Task 1 even though the replay buffer was empty. Deterministic algorithms were then enabled, after which the two trajectories matched as expected. A separate EWC run with lambda 1,000 diverged at Task 5 and produced non-finite loss; the run was rejected, and the training loop now fails immediately on non-finite values.

Important remaining limitations are:

- the full Naive/Replay/EWC method comparison is still seed-42 only;
- the buffer currently stores already augmented float32 tensors, so old images do not receive fresh stochastic augmentation during replay;
- hyperparameters were not selected with a separate validation stream; and
- the three-seed capacity study estimates variability but does not establish statistical significance.

## 5. Next research question

The next method-level experiment will fix the buffer at 2,000 examples and compare the current uniform replay baseline with one curriculum- or importance-aware sampling rule. This isolates whether which examples are replayed changes retention and positive transfer beyond merely storing more data, motivated by work on replay curricula and dynamic continual data selection. The comparison will keep the seeds, replay batch size, and total training budget paired. DER++ remains a useful secondary baseline, and a complementary systems experiment will compare the current float32 tensor memory with compact uint8 source-image storage and fresh replay-time augmentation.

## References

1. A. Krizhevsky. *Learning Multiple Layers of Features from Tiny Images*. Technical report, 2009.
2. K. He, X. Zhang, S. Ren, and J. Sun. *Deep Residual Learning for Image Recognition*. CVPR, 2016.
3. J. Kirkpatrick et al. *Overcoming Catastrophic Forgetting in Neural Networks*. PNAS, 2017.
4. J. S. Vitter. *Random Sampling with a Reservoir*. ACM Transactions on Mathematical Software, 1985.
5. A. Chaudhry et al. *Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence*. ECCV, 2018.
6. R. J. Tee and M. Zhang. *Integrating Curricula with Replays: Its Effects on Continual Learning*. AAAI Symposium Series, 2023.
7. A. Maharana et al. *Adapt-∞: Scalable Continual Multimodal Instruction Tuning via Dynamic Data Selection*. ICLR, 2025.
