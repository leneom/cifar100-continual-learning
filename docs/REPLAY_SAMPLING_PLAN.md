# Replay sampling: uniform vs loss-prioritized (pre-registration)

Written 2026-09-29, before any run of this comparison.

## Question

With the replay buffer fixed at 2,000 examples, does drawing replay examples in
proportion to their recent loss reduce forgetting compared with uniform draws?
The earlier ablation asked *how many* examples to store; this asks *which*
stored examples to rehearse.

## Arms

Shared protocol: CIFAR-100, 10 tasks x 10 classes, single-head ResNet-18,
10 epochs per task, SGD (lr 0.1, momentum 0.9, weight decay 5e-4), batch 128,
replay batch 64, reservoir storage with capacity 2,000, deterministic CUDA,
WSL platform (`cifar-cl` environment).

| Arm | Draw policy |
|---|---|
| `uniform` | Uniform without replacement; bit-identical to the original `ReplayBuffer` draws. |
| `loss` | Prioritized replay (Schaul et al., 2016): slot priority `(last replay loss + 1e-3) ** 0.6`, weighted draws without replacement, new slots get the running maximum priority. No importance-weight correction. |

`alpha = 0.6` and `epsilon = 1e-3` are the PER defaults and are fixed before the
first run. Any later alpha sweep is exploratory and reported separately.

## Endpoints

- Primary: final average accuracy after task 10.
- Secondary: final average forgetting; per-task replay draw share and mean
  priority (`task_logs[*].replay_by_task`) as mechanism diagnostics.
- Statistics: paired by seed (`loss - uniform`), mean difference with paired
  Student-t 95% CI and exact sign-flip p value.

## Staging and decision rule

1. Stage 1: seeds 0-2 for both arms (6 runs) to estimate the effect size.
2. Stage 2: extend both arms to seeds 0-7 unless stage 1 shows an effect larger
   than 3 pp in either direction. The Tee and Zhang work showed that four seeds
   produced a spurious -0.39 pp effect under similar seed noise, so a
   sub-3 pp stage-1 result is not interpreted on its own.
3. The conclusion is drawn from the eight-seed paired comparison. A null result
   is reported as a bound on the effect (the CI), not as a failed experiment.

## Commands

```bash
python run_replay_sampling.py --seeds 0 1 2
python run_replay_sampling.py --seeds 0 1 2 3 4 5 6 7
```

## Outcome (2026-09-29)

Stage 1 (seeds 0-2) gave -0.00 pp [-2.49, +2.49], so both arms were extended to
seeds 0-7 as planned. Eight-seed paired result, `loss - uniform`:

- final average accuracy +0.59 pp [-0.98, +2.17], sign-flip p = 0.40 (5/8 positive);
- final average forgetting -1.01 pp [-2.56, +0.53], p = 0.16.

No effect is detected. Mechanism: during task 10 the mean priority of tasks 1-8
is 0.058 versus 1.0 under uniform, i.e. buffer loss has collapsed because the
stored examples are memorized, and the priority mainly shifts draws to task 9
(share 11.3% -> 17.1%). A follow-up needs a priority signal that memorization
cannot mask.
