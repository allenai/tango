# Fine-tuning a language model

```{include} ../../../examples/train_lm/README.md
:start-after: <!-- start overview -->
:end-before: <!-- end overview -->
```

```{tip}
You can find the full code for this example on [GitHub](https://github.com/allenai/tango/tree/main/examples/train_lm).
```

```{tip}
To learn more about the Tango integrations that allow you to train a 6B parameter model or larger, see the [Training at Scale](/guides/training_at_scale) guide.
```

## Components

We'll need to write a step for tokenizing the data and preparing it for language model training.
All of the other steps we need are provided by Tango integrations.

So, create a file called `tokenize.py` with following contents:

```{literalinclude} ../../../examples/train_lm/tokenize.py
:language: py
```

## Configuration file

Next you'll need to create a configuration file that defines the experiment. Just copy over these contents into a file called `config.jsonnet`:


```{literalinclude} ../../../examples/train_lm/config.jsonnet
```

## Run it

Now we can run the experiment with:

```bash
tango run config.jsonnet -i tokenize.py -d /tmp/results
```
