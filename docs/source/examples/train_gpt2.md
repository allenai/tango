```{include} ../../../examples/train_gpt2/README.md
```

You can see this example on GitHub at https://github.com/allenai/tango/tree/main/examples/train_gpt2.

## Components

Create a file called `components.py` with following contents:

```{literalinclude} ../../../examples/train_gpt2/components.py
:language: py
```

## Config

Then create the experiment configuration file, `config.jsonnet`:

```{literalinclude} ../../../examples/train_gpt2/config.jsonnet
```

## Run it

Now we can run the experiment with:

```bash
tango run config.jsonnet -i components -d /tmp/workspace
```
