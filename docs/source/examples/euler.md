```{include} ../../../examples/euler/README.md
```

## Running the experiment

If you haven't already, clone the [tango repository](https://github.com/allenai/tango) and then
change directories into `examples/euler`.

You can then run the experiment with:

```bash
tango run euler_general.jsonnet -i complex_arithmetic -w workspace
```

This will leave its results in a subdirectory of `workspace/runs/` corresponding to the name of the run.
The output it prints should look something like this:
```
Starting new run comic-heron
Server started at http://localhost:8080/run/comic-heron
[step i_times_pi] ● Starting step "i_times_pi"...
[step i_times_pi] ✓ Finished step "i_times_pi"
[step cos] ● Starting step "cos"...
[step cos] ✓ Finished step "cos"
[step sin] ● Starting step "sin"...
[step sin] ✓ Finished step "sin"
[step pow_e] ✓ Found output for step "i_times_pi" in cache (needed by "pow_e")...
[step pow_e] ● Starting step "pow_e"...
[step pow_e] ✓ Finished step "pow_e"
[step i_times_sin] ✓ Found output for step "sin" in cache (needed by "i_times_sin")...
[step i_times_sin] ● Starting step "i_times_sin"...
[step i_times_sin] ✓ Finished step "i_times_sin"
[step sum] ✓ Found output for step "cos" in cache (needed by "sum")...
[step sum] ✓ Found output for step "i_times_sin" in cache (needed by "sum")...
[step sum] ● Starting step "sum"...
[step sum] ✓ Finished step "sum"
[step sub] ✓ Found output for step "sum" in cache (needed by "sub")...
[step sub] ✓ Found output for step "pow_e" in cache (needed by "sub")...
[step sub] ● Starting step "sub"...
[step sub] ✓ Finished step "sub"
[step print] ✓ Found output for step "sub" in cache (needed by "print")...
[step print] ● Starting step "print"...
[step print] 0j
[step print] ✓ Finished step "print"
✓ Finished run comic-heron

 ┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
 ┃ Step Name   ┃ Status      ┃ Cached Result                                                     ┃
 ┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
 │ cos         │ ✓ succeeded │ workspace/cache/CosineStep-5aes9CUTRmkz5gJ5J6JSRbJZ4qkFu4kk       │
 │ i_times_pi  │ ✓ succeeded │ workspace/cache/MultiplyStep-4SRzHCCqYGs2PLeT8LeL5ukrCWGJoiae     │
 │ i_times_sin │ ✓ succeeded │ workspace/cache/MultiplyStep-2ZG7wPj9WLn5PgpYyPVHw9Qg7VM1mhwf     │
 │ pow_e       │ ✓ succeeded │ workspace/cache/ExponentiateStep-1swPpNipP6HBSP5rKdNjEqbYAWNf4CdG │
 │ print       │ ✓ succeeded │ N/A                                                               │
 │ sin         │ ✓ succeeded │ workspace/cache/SineStep-5aes9CUTRmkz5gJ5J6JSRbJZ4qkFu4kk         │
 │ sub         │ ✓ succeeded │ workspace/cache/SubtractionStep-4ygj1UyLk6TCVBxN7DWTCccbMa7M1C5v  │
 │ sum         │ ✓ succeeded │ workspace/cache/AdditionStep-34AiXoyiPKADMUnhcBzFYd6JeMcgx4DP     │
 └─────────────┴─────────────┴───────────────────────────────────────────────────────────────────┘
                                                                 ✓ 8 succeeded

Use your workspace to get the cached result of a step, e.g.

 >>> from tango import Workspace
 >>> workspace = Workspace.from_url(...)
 >>> workspace.step_result_for_run("comic-heron", "sum")
```

A few things are of note here:
 1. Tango assigns a name to your run. In this case, the name is "comic-heron".
 2. In this configuration, the "print" step prints the output ("`0j`"). Most of the time though, you will look
    for the output in the output directories that are given in the table.
 3. You might notice that the "print" step produces no output. That's because it is uncacheable, and thus writes
    out nothing.


## Change a step

Let's make an update to a step! Open `complex_arithmetic.py` and change `AdditionStep`. The actual change you make
in the `run()` method does not matter, but the important thing is to update the `VERSION` member of the
`AdditionStep` class. `AdditionStep` does not yet have a `VERSION`, so we will give it one:
```Python
@Step.register("cadd")
class AdditionStep(Step):
    VERSION = "002"     # This is the important change.
    
    def run(self, a: ComplexOrTuple, b: ComplexOrTuple) -> complex:  # type: ignore
        return make_complex(a) + make_complex(b)
```

Now run the config again with
```bash
tango run euler_general.jsonnet -i complex_arithmetic -w workspace
```

This time, the output will look like this:
```
Starting new run right-amoeba
Server started at http://localhost:8080/run/right-amoeba
[step sum] ✓ Found output for step "cos" in cache (needed by "sum")...
[step sum] ✓ Found output for step "i_times_sin" in cache (needed by "sum")...
[step sum] ● Starting step "sum"...
[step sum] ✓ Finished step "sum"
[step sub] ✓ Found output for step "sum" in cache (needed by "sub")...
[step sub] ✓ Found output for step "pow_e" in cache (needed by "sub")...
[step sub] ● Starting step "sub"...
[step sub] ✓ Finished step "sub"
[step print] ✓ Found output for step "sub" in cache (needed by "print")...
[step print] ● Starting step "print"...
[step print] 0j
[step print] ✓ Finished step "print"
✓ Finished run right-amoeba

 ┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
 ┃ Step Name   ┃ Status      ┃ Cached Result                                                     ┃
 ┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
 │ cos         │ - not run   │ workspace/cache/CosineStep-5aes9CUTRmkz5gJ5J6JSRbJZ4qkFu4kk       │
 │ i_times_pi  │ - not run   │ workspace/cache/MultiplyStep-4SRzHCCqYGs2PLeT8LeL5ukrCWGJoiae     │
 │ i_times_sin │ - not run   │ workspace/cache/MultiplyStep-2ZG7wPj9WLn5PgpYyPVHw9Qg7VM1mhwf     │
 │ pow_e       │ - not run   │ workspace/cache/ExponentiateStep-1swPpNipP6HBSP5rKdNjEqbYAWNf4CdG │
 │ print       │ ✓ succeeded │ N/A                                                               │
 │ sin         │ - not run   │ workspace/cache/SineStep-5aes9CUTRmkz5gJ5J6JSRbJZ4qkFu4kk         │
 │ sub         │ ✓ succeeded │ workspace/cache/SubtractionStep-42mdcQBtrNAYvxYhmzdd1vj2uCG8N5Yf  │
 │ sum         │ ✓ succeeded │ workspace/cache/AdditionStep-002-34AiXoyiPKADMUnhcBzFYd6JeMcgx4DP │
 └─────────────┴─────────────┴───────────────────────────────────────────────────────────────────┘
                                                           ✓ 3 succeeded, 5 not run

Use your workspace to get the cached result of a step, e.g.

 >>> from tango import Workspace
 >>> workspace = Workspace.from_url(...)
 >>> workspace.step_result_for_run("right-amoeba", "sum")
```

As you can see, it re-used the cached results for several of the steps, and only ran three steps anew.

```{eval-rst}
:class:`tango.step.Step.VERSION` is just one of the ways in which you can change the behavior of a step. Head over to the
documentation of the :class:`tango.step.Step` class to see the others.
```
