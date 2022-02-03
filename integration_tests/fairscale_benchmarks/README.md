# FairScale Benchmarks

This integration test is for checking the performance of the `FairScaleTrainingEngine` with various configurations.

**When to run it:** It should be ran every time there is a major PyTorch or FairScale upgrade.

**Where to run it:** A server with 4 A100 GPUs. Make sure you set your `WANDB_API_KEY` environment variable.

**How to run it:** From the root directory of this repository, run:
```
tango run integration_tests/fairscale_benchmarks/config.jsonnet -i examples/train_lm/tokenize_step.py
```
