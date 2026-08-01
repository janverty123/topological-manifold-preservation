# EXP-001: Split-MNIST 2-Task Baseline

**Status:** validated
**Config:** `configs/baseline/split_mnist_2task.yaml`
**Related:** `EXPERIMENT_LOG.md#EXP-001`

## Objective

Establish the simplest possible test of TMP: 2-task Split-MNIST
(digits 0-4, then 5-9), comparing Finetune vs. EWC vs. TMP, per the
approved research plan's Methodology section.

## How to reproduce

```bash
python scripts/run_finetune.py --config configs/baseline/split_mnist_2task.yaml
python scripts/run_ewc.py --config configs/baseline/split_mnist_2task.yaml
python scripts/run_tmp.py --config configs/baseline/split_mnist_2task.yaml
python compare_results.py --config configs/baseline/split_mnist_2task.yaml
```

To reproduce the lambda validation:
```bash
python sweep_tmp_lambda.py --config configs/baseline/split_mnist_2task.yaml
```

## Results summary

| Method | Retention accuracy (final) | Learning accuracy (final) |
|---|---|---|
| finetune | 24.15% | 98.3% |
| ewc (lambda=20000) | 68.87% | 98.05% |
| tmp (lambda=5.0) | 96.28% | 98.81% |

`lambda_=0.0` ablation confirmed retention collapses to ~24% (matching
Finetune), proving the TMP regularizer itself -- not incidental code
structure -- drives the improvement.

## Conclusion

TMP substantially outperforms both baselines on this 2-task setup.
This result is stable and was reached only after two real bugs were
found and fixed along the way (see EXPERIMENT_LOG.md#EXP-001):
1. `ewc.lambda_` needed retuning from 400 -> 20000 after diagnosing a
   tiny Fisher magnitude.
2. TMP's differentiable surrogate initially compared the WRONG
   activations (current Task-2 batch vs. Task-1 baseline -- meaningless,
   since they're different images); fixed to compare a FIXED Task-1
   reference set against itself over time.

## Next steps

Generalize to more tasks within Split-MNIST to stress-test whether
protection holds up over a longer sequence -> EXP-002.
