# Overview

## Steps

Tango is a Python library for choreographing machine learning research experiments by executing
a series of steps.
A step can do anything, really, such as [prepare a dataset](tango.integrations.datasets.LoadDataset), [train a model](tango.integrations.torch.TorchTrainStep), send an email to your mother wishing her happy birthday, *etc*.

Concretely, each step is just a subclass of {class}`~tango.step.Step`, where the {meth}`~tango.step.Step.run` method in particular defines what the step actually does.
So anything that can be implemented in Python can be ran as a step.

Steps can also depend on other steps in that the output of one step can be (part of) the input to another step.
Thefore the steps that make up an experiment form a [directed graph](tango.step_graph.StepGraph).

The concept of the {class}`~tango.step.Step` is the bread and butter that makes Tango general and powerful.
So powerful, in fact, that you might be wondering if Tango is [Turing-complete](https://en.wikipedia.org/wiki/Turing_completeness)?
Well, we don't know yet, but we can say at least that Tango is **Tango-complete**.

```{note}
Definition: **Tango-complete** | adj.

*A system is **Tango-complete** if it can simulate Tango.*
```

## Configuration files

Experiments themselves are defined through JSON or [Jsonnet](https://jsonnet.org/) configuration files.
At a minimum, these files must contain the "steps" field, which should be a mapping of arbitrary (yet unique) step names to the configuration of the corresponding step.

For example:

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
    }
  }
}
```

*Can you guess what this experiment does?*

There are two steps in this experiment graph: "random_name" is the name of one step and "say_hello" is the name of the other.
The "type" parameter within the config of each *step* tells Tango which {class}`~tango.step.Step` class implementation to use for that step.

So, within the "random_name" step config

```json
"random_name": {
  "type": "random_choice",
  "choices": ["Turing", "Tango", "Larry"],
}
```

the `"type": "random_choice"` part tells Tango to use the {class}`~tango.step.Step` subclass that is registered by the name "random_choice".

But wait... what do I mean by *registered*?
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

Now back to our example.
The step classes referenced in our configuration file ("random_choice" and "concat_strings") don't actually exist in the Tango library,
but we can easily implement and register them in on our own.

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

## The executor

## The step cache

## What Tango is NOT

Tango is a not library that ties you to any specific framework, such as PyTorch
