# Train GPT2

This Tango example showcases how you could fine-tune GPT2 from [transformers](https://github.com/huggingface/transformers) on WikiText2 or a similar dataset.
It's best that you run this experiment on a machine with a GPU and PyTorch [properly installed](https://pytorch.org/get-started/locally/#start-locally),
otherwise Tango will fall back to CPU-only and it will be extremely slow.

You can kick off a training job with the following command:

```
$ tango run config.jsonnet
```
