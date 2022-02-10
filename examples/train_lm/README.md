# Fine-tuning a language model

<!-- start overview -->

This Tango example showcases how you could train or fine-tune a causal language model like [GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2)
or [GPT-J](https://huggingface.co/docs/transformers/model_doc/gptj) from [transformers](https://github.com/huggingface/transformers) on WikiText2 or a similar dataset.
It's best that you run this experiment on a machine with a GPU and PyTorch [properly installed](https://pytorch.org/get-started/locally/#start-locally), otherwise Tango will fall back to CPU-only and it will be extremely slow.

This example also depends on [FairScale](https://fairscale.readthedocs.io/en/latest/), which allows you to leverage [`FullyShardedDataParallel`](https://fairscale.readthedocs.io/en/latest/api/nn/fsdp.html) (FSDP) and [activation checkpointing](https://fairscale.readthedocs.io/en/latest/api/nn/checkpoint/checkpoint_activations.html) to fine-tune [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B) or a similar-sized model. Just set the constants `fsdp` and `activation_checkpointing` in the config to `true`.
Without using CPU offloading you'll need at least 4 x 40GiB A100 GPUs, or a different configuration with a comparable amount of total GPU memory.

<!-- end overview -->

To getting started, just run

```
tango run config.jsonnet -i tokenize_step.py
```
