```{include} ../../../examples/train_gpt2/README.md
```

## Getting started

### Dependencies

To get started, first install the following dependencies:

```{literalinclude} ../../../examples/train_gpt2/requirements.txt
```

### Components

Next create a file called `components.py` with following contents:

```{literalinclude} ../../../examples/train_gpt2/components.py
:language: py
```

### Config

Then create the experiment configuration file, `config.jsonnet`:

```{literalinclude} ../../../examples/train_gpt2/config.jsonnet
```

### Run it

Now we can run the experiment with:

```bash
tango run config.jsonnet -i components -d /tmp/results
```
