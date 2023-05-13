# First Steps

## What is a Step?

Tango is a Python library for choreographing machine learning research experiments by executing
a series of steps.
A step can do anything, really, such as [prepare a dataset](tango.integrations.datasets.LoadDataset), [train a model](tango.integrations.torch.TorchTrainStep), send an email to your mother wishing her happy birthday, *etc*.

Concretely, each step is just a subclass of {class}`~tango.step.Step`, where the {meth}`~tango.step.Step.run` method in particular defines what the step actually does.
So anything that can be implemented in Python can be run as a step.

Steps can also depend on other steps in that the output of one step can be part of the input to another step.
Therefore, the steps that make up an experiment form a [directed graph](tango.step_graph.StepGraph).

The concept of the {class}`~tango.step.Step` is the bread and butter that makes Tango so general and powerful.
*So* powerful, in fact, that you might be wondering if Tango is [Turing-complete](https://en.wikipedia.org/wiki/Turing_completeness)?
Well, we don't know yet, but we can say at least that Tango is **Tango-complete** 😉

## Configuration files

Experiments themselves are defined through JSON, [Jsonnet](https://jsonnet.org/), or YAML configuration files.
At a minimum, these files must contain the "steps" field, which should be a mapping of arbitrary (yet unique) step names to the configuration of the corresponding step.

For example, let's create a config file called `config.jsonnet` with the following contents:

```json
{
  "steps": {
    "random_name": {
      "type": "random_choice",
      "choices": ["Turing", "Tango", "Larry"],
    },
    "say_hello": {
      "type": "concat_strings",
      "string1": "Hello, ",
      "string2": {
        "type": "ref",
        "ref": "random_name"
      }
    },
    "print": {
      "type": "print",
      "input": {
        "type": "ref",
        "ref": "say_hello"
      }
    }
  }
}
```

*Can you guess what this experiment does?*

There are three steps in this experiment graph: "random_name" is the name of one step, "say_hello" is the name of another, and "print" is the name of the last.
The "type" parameter within the config of each step tells Tango which {class}`~tango.step.Step` class implementation to use for that step.

So, within the "random_name" step config

```json
"random_name": {
  "type": "random_choice",
  "choices": ["Turing", "Tango", "Larry"],
}
```

the `"type": "random_choice"` part tells Tango to use the {class}`~tango.step.Step` subclass that is registered by the name "random_choice".

But wait... what do we mean by *registered*?

Tango keeps track of an internal registry for certain classes (such as the {class}`~tango.step.Step` class) that is just a mapping of arbitrary unique names to subclasses.
When you look through Tango's source code, you'll see things like:

```python
@Step.register("foo")
class Foo(Step):
    ...
```

This is how subclasses get added to the registry.
In this case the subclass `Foo` is added to the `Step` registry under the name "foo", so if you were to use `"type": "foo"` in your configuration file, Tango would understand
that you mean to use the `Foo` class for the given step.

```{tip}
Any class that inherits from {class}`~tango.common.registrable.Registrable` can have its own
registry.
```

Now back to our example.
The step classes referenced in our configuration file ("random_choice" and "concat_strings") don't actually exist in the Tango library (though the ["print" step](tango.steps.PrintStep) does),
but we can easily implement and register them on our own.

Let's put them in a file called `components.py`:

```python
# file: components.py

import random
from typing import List

from tango import Step

@Step.register("random_choice")
class RandomChoiceStep(Step):
    DETERMINISTIC = False

    def run(self, choices: List[str]) -> str:
        return random.choice(choices)

@Step.register("concat_strings")
class ConcatStringsStep(Step):
    def run(self, string1: str, string2: str) -> str:
        return string1 + string2
```

```{important}
It's important that you use type hints in your code so that Tango can properly construct Python objects from the corresponding serialized (JSON) objects
and warn you when the types don't match up.
```

So as long as Tango is able to import this module (`components.py`) these step implementations will be added to the registry
and Tango will know how to instantiate and run them.

There's also a short-hand way of implementing steps, using the {func}`@step() <tango.step.step>` function decorator:

```python
from tango import step

@step(deterministic=False)
def random_choice(choices: List[str]) -> str:
    return random.choice(choices)

@step()
def concat_strings(string1: str, string2: str) -> str:
    return string1 + string2
```

This will register these steps under the name of the corresponding function, i.e. "random_choice" and "concat_strings", by default, though that can be overridden by specifying the "name" parameter to the decorator:

```python
@step(name="random-string", deterministic=False)
def random_choice(choices: List[str]) -> str:
    return random.choice(choices)
```

## Executing an experiment

At this point we've implemented our custom steps (`components.py`) and created our configuration
file `config.jsonnet`, so we're ready to actually run this experiment.

For that, just use the `tango run` command:

```
$ tango run config.jsonnet -i components
```

```{tip}
- The `-i` option is short for `--include-package`, which takes the name of a Python package which Tango will try to import.
In this case our custom steps are in `components.py`, so we need Tango to import this module to find those steps.
As long as `components.py` is in the current directory or somewhere else on the `PYTHONPATH`, Tango will be able to find and import
this module when you pass `-i components` (note the lack of the `.py` at the end).
```

You should see something like this in the output:

```
Starting new run cute-kitten
● Starting step "random_name"
✓ Finished step "random_name"
● Starting step "say_hello"
✓ Finished step "say_hello"
● Starting step "print"
Hello, Tango
✓ Finished step "print"
```

## Step caching

This particular experiment didn't write any results to disk, but in many situations you'll want to save the output of at least some of your steps.

For example, if you're using the {class}`~tango.integrations.torch.TorchTrainStep` step, the output is a trained model, which is certainly a useful thing to keep around.
In other cases, you may not actually care about the direct result of a particular step, but it could still be useful to save it when possible so that Tango doesn't need to run the step
again unnecessarily.

This is where Tango's caching mechanism comes in.

To demonstrate this, let's look at another example that pretends to do some expensive computation.
Here is the `config.jsonnet` file:

```json
{
  "steps": {
    "add_numbers": {
      "type": "really_inefficient_addition",
      "num1": 34,
      "num2": 8
    }
  }
}
```

And let's implement "really_inefficient_addition":

```python
# components.py

import time

from tango import Step, JsonFormat
from tango.common import Tqdm


@Step.register("really_inefficient_addition")
class ReallyInefficientAdditionStep(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = JsonFormat()

    def run(self, num1: int, num2: int) -> int:
        for _ in Tqdm.tqdm(range(100), desc="Computing...", total=100):
            time.sleep(0.05)
        return num1 + num2
```

There are a couple of things to note about this step, other than the obvious inefficiencies; the class variables
we've defined: {attr}`~tango.step.Step.DETERMINISTIC`, {attr}`~tango.step.Step.CACHEABLE`, and
{attr}`~tango.step.Step.FORMAT`.

`DETERMINISTIC = True` tells Tango that, given particular inputs, the output to this step will always be the same
every time it is ran, which has implications on caching.
By default, Tango assumes steps are deterministic.
You can override this by saying `DETERMINISTIC = False`.
Tango will warn you when you try to cache a non-deterministic step.

`CACHEABLE = True` tells Tango that it can cache this step and `FORMAT = JsonFormat()` defines which
{class}`~tango.format.Format` Tango will use to serialize the result of the step.

This time when we run the experiment we'll designate a specific directory for Tango to use:

```bash
$ tango run config.jsonnet -i components -d workspace/
```
```
Starting new run live-tarpon
● Starting step "add_numbers"
Computing...: 100%|##########| 100/100 [00:05<00:00, 18.99it/s]
✓ Finished step "add_numbers"
✓ The output for "add_numbers" is in workspace/runs/live-tarpon/add_numbers
```

The last line in the output tells us where we can find the result of our "add_numbers" step. `live-tarpon` is
the name of the run. Run names are randomly generated and may be different on your machine. `add_numbers` is the
name of the step in your config. The whole path is a symlink to a directory, which contains (among other things)
a file `data.json`:

```bash
$ cat workspace/runs/live-tarpon/add_numbers/data.json
```
```
42
```

Now look what happens when we run this step again:

```bash
$ tango run config.jsonnet -i components -d workspace/
```
```
Starting new run modest-shrimp
✓ Found output for "add_numbers" in cache
✓ The output for "add_numbers" is in workspace/runs/modest-shrimp/add_numbers
```

Tango didn't have to run our really inefficient addition step this time because it found the previous cached
result. It put the results in the result directory for a different run (in our case, the `modest-shrimp` run),
but once again it is a symlink that links to the same results from our first run.

If we changed the inputs to the step in `config.jsonnet`:

```diff
     "add_numbers": {
       "type": "really_inefficient_addition",
       "num1": 34,
-      "num2": 8
+      "num2": 2
     }
   }
 }
```

And ran it again:

```bash
$ tango run config.jsonnet -i components -d workspace/
```
```
Starting new run true-parrot
● Starting step "add_numbers"
Computing...: 100%|##########| 100/100 [00:05<00:00, 19.13it/s]
✓ Finished step "add_numbers"
✓ The output for "add_numbers" is in workspace/runs/true-parrot/add_numbers
```

You'd see that Tango had to run our "add_numbers" step again.

You may have noticed that `workspace/runs/true-parrot/add_numbers` is now a symlink that points to a different
place than it did for the first two runs. That's because it produced a different result this time. All the
result symlinks point into the `workspace/cache/` directory, where all the step's results are cached.

This means that if we ran the experiment again with the original inputs, Tango would still find the cached result
and wouldn't need to rerun the step.

## Arbitrary objects as inputs

### `FromParams`

So far the inputs to all of the steps in our examples have been built-in Python types that can be deserialized from JSON (e.g. {class}`int`, {class}`str`, etc.),
but sometimes you need the input to a step to be an instance of an arbitrary Python class.

Tango allows this as well as it can infer from type hints what the class is and how to instantiate it.
When writing your own classes, it's recommended that you have your class inherit from the {class}`~tango.common.from_params.FromParams` class, which will gaurantee that
Tango can instantiate it from a config file.

For example, suppose we had a step like this:

```python
from tango import Step
from tango.common import FromParams


class Bar(FromParams):
    def __init__(self, x: int) -> None:
        self.x = x


@Step.register("foo")
class FooStep(Step):
    def run(self, bar: Bar) -> int:
        return bar.x
```

```{tip}
If you've used [AllenNLP](https://github.com/allenai/allennlp) before, this will look familiar!
In fact, it's the same system under the hood.
```

Then we could create a config like this:

```json
{
  "steps": {
    "foo": {
      "type": "foo",
      "bar": {"x": 1}
    }
  }
}
```

And Tango will figure out how to deserialize `{"x": 1}` into a `Bar` instance.

You can also have `FromParams` objects nested within other `FromParams` objects or standard containers
like {class}`list`:

```python
from typing import List

from tango import Step
from tango.common import FromParams


class Bar(FromParams):
    def __init__(self, x: int) -> None:
        self.x = x


class Baz(FromParams):
    def __init__(self, bar: Bar) -> None:
        self.bar = bar


@Step.register("foo")
class FooStep(Step):
    def run(self, bars: List[Bar], baz: Baz) -> int:
        return sum([bar.x for bar in bars]) + baz.bar.x
```

### `Registrable`

The {class}`~tango.common.registrable.Registrable` class is a special kind of {class}`~tango.common.from_params.FromParams` class that allows you to specify from the config which subclass of an expected class to deserialize into.

This is actually how we've been instantiating specific `Step` subclasses. Because {class}`~tango.step.Step` inherits from {class}`~tango.common.registrable.Registrable`, we can use the `"type"` fields in the config file to specify a `Step` subclass.

This is also very useful when you're writing a step that requires a certain type as input, but you want to be able to change the exact subclass of the type from your config file. For example, the {class}`~tango.integrations.torch.TorchTrainStep` takes `Registrable` inputs such as {class}`~tango.integrations.torch.Model`. Model variants can then be subclasses that are specified in the config file by their registered names. A sketch of this might look like the following: 

```python
from tango import Step
from tango.common import FromParams, Registrable

class Model(torch.nn.Module, Registrable):
    ...

@Model.register("variant1")
class Variant1(Model):
    ...

@Model.register("variant2")
class Variant2(Model):
    ...

@Step.register("torch::train")
class TorchTrainerStep(Step):
    def run(self, model: Model, ...) -> Model:
        ...
```

And a sketch of the config file would be something like this:

```json
{
  "steps": {
    "train": {
      "type": "torch::train",
      "model": {
        "type": "variant1",
      }
    }
  }
}
```

As in the `FromParams` example the specifications can be nested, but now we also denote the subclass with the `"type": "..."` field. To swap models we need only change "variant1" to "variant2" in the config. The value for "type" can either be the name that the class is registered under (e.g. "train" for `TorchTrainStep`), or the fully qualified class name (e.g. `tango.integrations.torch.TorchTrainStep`).

You'll see more examples of this in the [next section](examples/index).
