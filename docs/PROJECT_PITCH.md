# Project Pitch

## 125-word version

I built a fully reproducible class-incremental learning pipeline to study catastrophic forgetting on CIFAR-100. A ResNet-18 learns ten sequential tasks through a single 100-class output head and is evaluated on every observed task after each stage. I implemented Naive Fine-Tuning, reservoir-sampling Experience Replay, and Online EWC, with deterministic comparisons, accuracy matrices, tests, and guards for divergent runs. In the seed-42 baseline, Replay with 2,000 examples reached 17.96% final average accuracy, compared with 6.64% for Naive and 6.37% for EWC. I then ran a buffer-capacity ablation and three-seed validation. Increasing the buffer from 1,000 to 5,000 raised final accuracy from 12.14% to 26.79% while reducing forgetting from 60.56% to 42.63%. My next question is whether curriculum-aware replay sampling further improves this accuracy-memory trade-off.

## 40-word version

I built and validated a CIFAR-100 continual-learning pipeline comparing Naive Fine-Tuning, Experience Replay, and Online EWC. Three-seed experiments show that larger replay buffers consistently improve accuracy and reduce forgetting. I am now studying selective and curriculum-aware replay at fixed capacity.

## One-sentence version

I study how replay design affects catastrophic forgetting in single-head class-incremental image classification, using deterministic experiments, complete accuracy matrices, and multi-seed validation on CIFAR-100.

## Conversation opener

I have been working on a small class-incremental learning study on CIFAR-100. The clearest result is that Replay strongly outperforms Naive and EWC in my current single-head setup, and the effect of buffer capacity is consistent across three seeds. I would be interested in learning how your group approaches replay bias, representation drift, or continual adaptation in more realistic settings.
