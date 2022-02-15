```{include} ../../../examples/eval_p3/README.md
```

## `RougeScoreStep`

`RougeScoreStep` is defined in `eval.py`:

```{literalinclude} ../../../examples/eval_p3/eval.py
:language: py
```

## Config

The configuration file, `config.jsonnet`, uses some advanced [Jsonnet](https://jsonnet.org) concepts like `std.foldl`
to create the same configuration for all 10 prompts:

```{literalinclude} ../../../examples/eval_p3/config.jsonnet
```

## Run it

You can run the experiment with:

```bash
tango run config.jsonnet -i eval -d /tmp/workspace
```
