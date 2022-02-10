# FairScale Benchmarks

This integration test is for checking the performance of the `FairScaleTrainingEngine` with various configurations.

**When to run it:** It should be ran every time there is a major PyTorch or FairScale upgrade.

**Where to run it:** A server with 4 A100 GPUs. Make sure you set your `WANDB_API_KEY` environment variable.

**How to run it:** From the root directory of this repository, run:
```
integration_tests/fairscale_benchmarks/run.sh
```

**What to look for:** The training jobs shouldn't fail, for one. After `tango run` completes, check the corresponding Weights & Biases
dashboard and inspect the results. Compare the various "fsdp" training runs with the baseline to ensure you see memory savings.
