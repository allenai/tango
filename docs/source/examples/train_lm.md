# Fine-tuning a language model

```{include} ../../../examples/train_lm/README.md
:start-after: <!-- start overview -->
:end-before: <!-- end overview -->
```

```{tip}
You can find the full code for this example on [GitHub](https://github.com/allenai/tango/tree/main/examples/train_lm).
```

## Components

Create a file called `components.py` with following contents:

```{literalinclude} ../../../examples/train_lm/components.py
:language: py
```

## Configuration file

Next you'll need to create a configuration file that defines the experiment. Just copy over these contents into a file called `config.jsonnet`:


```{literalinclude} ../../../examples/train_lm/config.jsonnet
```

## Run it

Now we can run the experiment with:

```bash
tango run config.jsonnet -i components -d /tmp/results
```
