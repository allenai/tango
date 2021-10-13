# Overview

## Steps

Tango is a Python library for choreographing machine learning research experiments by executing
a series of steps.
A step can do anything, really, such as [prepare a dataset](tango.integrations.datasets.LoadDataset), [train a model](tango.integrations.torch.TorchTrainStep), send an email to your mother wishing her happy birthday, *etc*.

Concretely, each step is just a subclass of {class}`~tango.step.Step`, where the {meth}`~tango.step.Step.run` method in particular defines what the step actually does.
So anything that can be implemented in Python can be ran as a step.

Steps can also depend on other steps in that the output of one step can be (part of) the input to another step.
Thefore the steps that make up an experiment form a [directed graph](tango.step_graph.StepGraph).

The concept of the {class}`~tango.step.Step` is the bread and butter that makes Tango so general and powerful.
*So* powerful, in fact, that you might be wondering if Tango is [Turing-complete](https://en.wikipedia.org/wiki/Turing_completeness)?
Well, we don't know yet, but we can say at least that Tango is **Tango-complete** 😉

```{note}
Definition: **Tango-complete** | adj.

*A system is **Tango-complete** if it can simulate Tango.*
```

## Configuration files

Experiments themselves are defined through JSON or [Jsonnet](https://jsonnet.org/) configuration files.
At a minimum, these files must contain the "steps" field, which should be a mapping of arbitrary (yet unique) step names to the configuration of the corresponding step.

For example, let's create a config file called `config.jsonnet` with following contents:

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
The "type" parameter within the config of each *step* tells Tango which {class}`~tango.step.Step` class implementation to use for that step.

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
It's important that you use type hints in your code so that Tango can properly construct Python objects from the corresponding JSON objects
and warn you when the types don't match up.
```

So as long as Tango is able to import this module (`components.py`) these step implementations will be added to the registry
and Tango will know how to instantiate and run them.

## Executing an experiment

At this point we've implemented our custom steps (`components.py`) and created our configuration
file `config.jsonnet`, so we're ready to actually run this experiment.

For that, just use the `tango run` command:

```
$ tango --no-logging run config.jsonnet -i components
```

```{tip}
The `--no-logging` flag just suppresses logging, which is convenient for this example so we
can clearly see everything printed to stdout.
```

You should see something like this in the output:

```
● Starting run for "random_name"
✓ Finished run for "random_name"
● Starting run for "say_hello"
✓ Finished run for "say_hello"
● Starting run for "print"
Hello, Tango
✓ Finished run for "print"
```

## Step caching

This particular experiment didn't write any results to disk, but in many situations you'll want to save the output of at least some of your steps.

For example, if you're using the {class}`~tango.integrations.torch.TorchTrainStep` step, the output is a trained model, which is certianly a useful thing to keep around.
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

There's a couple of odd things about this step, other than the obvious inefficiencies.
Specifically I'm talking about the class variables we've defined: {attr}`~tango.step.Step.DETERMINISTIC`, {attr}`~tango.step.Step.CACHEABLE`, and {attr}`~tango.step.Step.FORMAT`.
These are all meaningful to Tango.

`DETERMINISTIC = True` tells Tango that, given particular inputs, the output to this step will always be the same every time it is ran, which has implications on caching.
By default Tango assumes steps are non-deterministic, and it will warn you when you try to cache a non-deterministic step.

`CACHEABLE = True` tells Tango that it can cache this step and `FORMAT = JsonFormat()` defines which {class}`~tango.format.Format` Tango will use to serialize the result of the step.

This time when we run the experiment we'll designate a specific directory for Tango to use:

```bash
$ tango --no-logging run config.jsonnet -i components -d /tmp/add
```
```
● Starting run for "add_numbers"
Computing...: 100%|##########| 100/100 [00:05<00:00, 18.99it/s]
✓ Finished run for "add_numbers"
✓ The output for "add_numbers" is in /tmp/add/add_numbers
```

The last line in the output tells us where we can find the result of our "add_numbers" step.
This will be in a file called `data.json`:

```bash
$ cat /tmp/add/add_numbers/data.json
```
```
42
```

Now look what happens when we run this step again:

```bash
$ tango --no-logging run config.jsonnet -i components -d /tmp/add
```
```
✓ Found output for "add_numbers" in cache
✓ The output for "add_numbers" is in /tmp/add/add_numbers
```

Tango didn't have to run our really inefficient addition step this time because it found the previous cached result.
But if we changed the inputs to the step in `config.jsonnet`:

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
$ tango --no-logging run config.jsonnet -i components -d /tmp/add
```
```
● Starting run for "add_numbers"
Computing...: 100%|##########| 100/100 [00:05<00:00, 19.13it/s]
✓ Finished run for "add_numbers"
✓ The output for "add_numbers" is in /tmp/add/add_numbers
```

You'd see that Tango had to run our "add_numbers" step again.

You may have also noticed that the output path for "add_numbers" (`/tmp/add/add_numbers`) is the same as it was before, even though we had to rerun the step.
But if you inspect the `/tmp/add` directory you'll find that `add_numbers` is actually a symlink to another directory in `/tmp/add/step_cache`.
The `/tmp/add/step_cache` directory has two sub-directories at this point: one for each run of the step.

This means that if we ran the experiment again with the original inputs, Tango would still find the cached result and wouldn't need to rerun the step.

## Arbitrary objects as inputs

### `FromParams`

So far the inputs to all of the steps in our examples have been built-in Python types that can be deserialized from JSON (e.g. {class}`int`, {class}`str`, etc.),
but sometimes you need the input to a step to be an instance of an arbitrary Python class.

Tango allows this as well as long the class is a subclass of {class}`~tango.common.from_params.FromParams`, in which case Tango will infer how to deserialize the instance from the config
by inspecting the type hints of the class constructor.

```{tip}
If you've used [AllenNLP](https://github.com/allenai/allennlp) before, this will look familiar!
In fact, it's the same system under the hood.
```

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

The {class}`~tango.common.registrable.Registrable` class is special kind of {class}`~tango.common.from_params.FromParams` class that allows you to deserialize
any subclass of the expected class from JSON.

We've already come across this, actually, because {class}`~tango.step.Step` inherits from `Registrable`, why is why Tango is able to instantiate arbitrary `Step` subclasses from a config.

This is very useful when you're writing a step that requires a certain type as input, but you want to be able to change the exact subclass of the type from your configuration file.
For example, the {class}`~tango.integrations.torch.TorchTrainStep` step takes several `Registrable` base classes as input, including {class}`~tango.integrations.torch.Model` and
{class}`~tango.integrations.torch.Optimizer`.

In order to specify which subclass to deserialize to, you just need to add the `"type": "..."` field to the corresponding section of the JSON/Jsonnet config.
The value for "type" can either be the name that the class is registered under (e.g. "torch::train" for `TorchTrainStep`), or the fully qualified class name (e.g. `tango.integrations.torch.TorchTrainStep`).

You'll see more examples of this in the [next section](examples/index).
