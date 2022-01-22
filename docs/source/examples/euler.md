```{include} ../../../examples/euler/README.md
```

## Running the experiment

Try running the experiment with:

```bash
tango run euler_general.jsonnet -i complex_arithmetic -d workspace
```

This will leave its results in a directory called `workspace/`.

The output it prints should look something like this:
```
Starting new run fluent-jay
Server started at http://localhost:8080/run/fluent-jay
● Starting step "cos" ...
✓ Finished step "cos"
● Starting step "i_times_pi" ...
✓ Finished step "i_times_pi"
● Starting step "sin" (needed by "i_times_sin") ...
✓ Finished step "sin"
● Starting step "i_times_sin" ...
✓ Finished step "i_times_sin"
✓ Found output for step "i_times_pi" in cache (needed by "pow_e") ...
● Starting step "pow_e" ...
✓ Finished step "pow_e"
✓ Found output for step "cos" in cache (needed by "sum") ...
✓ Found output for step "i_times_sin" in cache (needed by "sum") ...
● Starting step "sum" (needed by "sub") ...
✓ Finished step "sum"
✓ Found output for step "pow_e" in cache (needed by "sub") ...
● Starting step "sub" (needed by "print") ...
✓ Finished step "sub"
● Starting step "print" ...
0j
✓ Finished step "print"
✓ Found output for step "sin" in cache ...
✓ Found output for step "sub" in cache ...
✓ Found output for step "sum" in cache ...
✓ The output for "cos" is in /home/dirkg/tango/examples/euler/workspace/cache/CosineStep-5aes9CUTRmkz5gJ5J6JSRbJZ4qkFu4kk
✓ The output for "i_times_pi" is in /home/dirkg/tango/examples/euler/workspace/cache/MultiplyStep-4SRzHCCqYGs2PLeT8LeL5ukrCWGJoiae
✓ The output for "i_times_sin" is in /home/dirkg/tango/examples/euler/workspace/cache/MultiplyStep-2ZG7wPj9WLn5PgpYyPVHw9Qg7VM1mhwf
✓ The output for "pow_e" is in /home/dirkg/tango/examples/euler/workspace/cache/ExponentiateStep-Rf73w34zWJcBrQafpAkxDvXR4mq3MXC9
✓ The output for "sin" is in /home/dirkg/tango/examples/euler/workspace/cache/SineStep-5aes9CUTRmkz5gJ5J6JSRbJZ4qkFu4kk
✓ The output for "sub" is in /home/dirkg/tango/examples/euler/workspace/cache/SubtractionStep-YCdedqjmmd9GUFi96VzPXD5tAVho3CTz
✓ The output for "sum" is in /home/dirkg/tango/examples/euler/workspace/cache/AdditionStep-34AiXoyiPKADMUnhcBzFYd6JeMcgx4DP
Finished run fluent-jay
```

## Try the workspace UI

You can visualize the workspace by running

```bash
tango server -d workspace
```

This command will start up a web server that visualizes workspace directories. It first presents you with a list
of runs. Click through to your run, and you see a graph of the steps that you just ran:

![Step Graph](../../_static/step_graph.png)

You can click on the arrow next to each step to see more information.

## Change a step

Let's make an update to a step! Open `complex_arimetic.py` and change `AdditionStep`. The actual change you make
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
tango run euler_general.jsonnet -i complex_arithmetic -d workspace
```

This time, the output will look like this:
```
Starting new run rested-kitten
Server started at http://localhost:8080/run/rested-kitten
✓ Found output for step "cos" in cache ...
✓ Found output for step "i_times_pi" in cache ...
✓ Found output for step "i_times_sin" in cache ...
✓ Found output for step "pow_e" in cache ...
✓ Found output for step "cos" in cache (needed by "sum") ...
✓ Found output for step "i_times_sin" in cache (needed by "sum") ...
● Starting step "sum" (needed by "sub") ...
✓ Finished step "sum"
✓ Found output for step "pow_e" in cache (needed by "sub") ...
● Starting step "sub" (needed by "print") ...
✓ Finished step "sub"
● Starting step "print" ...
0j
✓ Finished step "print"
✓ Found output for step "sin" in cache ...
✓ Found output for step "sub" in cache ...
✓ Found output for step "sum" in cache ...
✓ The output for "cos" is in /home/dirkg/tango/examples/euler/workspace/cache/CosineStep-5aes9CUTRmkz5gJ5J6JSRbJZ4qkFu4kk
✓ The output for "i_times_pi" is in /home/dirkg/tango/examples/euler/workspace/cache/MultiplyStep-4SRzHCCqYGs2PLeT8LeL5ukrCWGJoiae
✓ The output for "i_times_sin" is in /home/dirkg/tango/examples/euler/workspace/cache/MultiplyStep-2ZG7wPj9WLn5PgpYyPVHw9Qg7VM1mhwf
✓ The output for "pow_e" is in /home/dirkg/tango/examples/euler/workspace/cache/ExponentiateStep-Rf73w34zWJcBrQafpAkxDvXR4mq3MXC9
✓ The output for "sin" is in /home/dirkg/tango/examples/euler/workspace/cache/SineStep-5aes9CUTRmkz5gJ5J6JSRbJZ4qkFu4kk
✓ The output for "sub" is in /home/dirkg/tango/examples/euler/workspace/cache/SubtractionStep-5T37NJraCyYai4oFofngY6m7iha6jqYw
✓ The output for "sum" is in /home/dirkg/tango/examples/euler/workspace/cache/AdditionStep-001-34AiXoyiPKADMUnhcBzFYd6JeMcgx4DP
Finished run rested-kitten
```

As you can see, it re-used the cached results for several of the steps, and only ran three steps anew.

```{eval-rst}
:class:`tango.step.Step.VERSION` is just one of the ways in which you can change the behavior of a step. Head over to the
documentation of the :class:`tango.step.Step` class to see the others.
```
