```{include} ../../../examples/train_lm/README.md
```

## Components

Create a file called `components.py` with following contents:

```{literalinclude} ../../../examples/train_lm/components.py
:language: py
```

## Config

### Basic config for GPT-2

Then create the experiment configuration file, `config.jsonnet`:

```{literalinclude} ../../../examples/train_lm/config.jsonnet
```

### Config with FairScale for GPT-J 6B

If you're interested in trying out the larger GPT-J 6B model, use the config instead:

```{literalinclude} ../../../examples/train_lm/config-fairscale.jsonnet
```

## Run it

Now we can run the experiment with:

```bash
tango run config.jsonnet -i components -d /tmp/results
```
