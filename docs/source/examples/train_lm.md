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

## Configuration files

Next you'll need to create a configuration file that defines the experiment. We provide a couple of alternatives below. Just choose one and copy over the contents into a file called `config.jsonnet` (or whatever else you want to call it).

### Basic config for GPT-2

Here is a config for training the base GPT-2 model. You should be able to run this even if you don't have a GPU.

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
