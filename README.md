<div align="center">
<br>
<img src="https://raw.githubusercontent.com/allenai/tango/main/docs/source/_static/tango_final_horizontal.png" width="600"/>
<br>
<br>
<p>
<!-- start tagline -->
AI2 Tango replaces messy directories and spreadsheets full of file versions by organizing experiments into discrete steps that can be cached and reused throughout the lifetime of a research project.
<!-- end tagline -->
</p>
<hr/>
<a href="https://github.com/allenai/tango/actions">
    <img alt="CI" src="https://github.com/allenai/tango/workflows/CI/badge.svg?event=push&branch=main">
</a>
<a href="https://pypi.org/project/ai2-tango/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/ai2-tango">
</a>
<a href="https://ai2-tango.readthedocs.io/en/latest/?badge=latest">
    <img src="https://readthedocs.org/projects/ai2-tango/badge/?version=latest" alt="Documentation Status" />
</a>
<a href="https://github.com/allenai/tango/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/allenai/tango.svg?color=blue&cachedrop">
</a>
<br/>
</div>

## Quick links

- [Documentation](https://ai2-tango.readthedocs.io/)
- [PyPI Package](https://pypi.org/project/ai2-tango/)
- [Contributing](https://github.com/allenai/tango/blob/main/CONTRIBUTING.md)
- [License](https://github.com/allenai/tango/blob/main/LICENSE)

## In this README

- [Quick start](#quick-start)
- [Installation](#installation)
  - [Installing with PIP](#installing-with-pip)
  - [Installing with Conda](#installing-with-conda)
  - [Installing from source](#installing-from-source)
  - [Checking your installation](#checking-your-installation)
  - [Docker image](#docker-image)
- [FAQ](#faq)
- [Team](#team)
- [License](#license)

## Quick start

Create a Tango step:

```python
# hello.py

from tango import step

@step()
def hello(name: str) -> str:
    message = f"Hello, {name}!"
    print(message)
    return message
```

And create a corresponding experiment configuration file:

```jsonnet
// hello.jsonnet

{
  steps: {
    hello: {
      type: "hello",
      name: "World",
    }
  }
}
```

Then run the experiment using a local workspace to cache the result:

```bash
tango run hello.jsonnet -w /tmp/workspace
```

You'll see something like this in the output:

```
Starting new run expert-llama
● Starting step "hello"...
Hello, World!
✓ Finished step "hello"
✓ Finished run expert-llama
```

If you run this a second time the output will now look like this:

```
Starting new run open-crab
✓ Found output for step "hello" in cache...
✓ Finished run open-crab
```

You won't see "Hello, World!" this time because the result of the step was found in the cache, so it wasn't run again.

For a more detailed introduction check out the [First Steps](https://ai2-tango.readthedocs.io/en/latest/first_steps.html) walk-through.

## Installation

<!-- start install -->

**ai2-tango** requires Python 3.8 or later.

### Installing with `pip`

**ai2-tango** is available [on PyPI](https://pypi.org/project/ai2-tango/). Just run

```bash
pip install ai2-tango
```

To install with a specific integration, such as `torch` for example, run

```bash
pip install 'ai2-tango[torch]'
```

To install with all integrations, run

```bash
pip install 'ai2-tango[all]'
```

### Installing with `conda`

**ai2-tango** is available on conda-forge. You can install just the base package with

```bash
conda install tango -c conda-forge
```

You can pick and choose from the integrations with one of these:

```bash
conda install tango-datasets -c conda-forge
conda install tango-torch -c conda-forge
conda install tango-wandb -c conda-forge
```

You can also install everything:

```bash
conda install tango-all -c conda-forge
```

Even though **ai2-tango** itself is quite small, installing everything will pull in a lot of dependencies.
Don't be surprised if this takes a while!

### Installing from source

To install **ai2-tango** from source, first clone [the repository](https://github.com/allenai/tango):

```bash
git clone https://github.com/allenai/tango.git
cd tango
```

Then run

```bash
pip install -e '.[all]'
```

To install with only a specific integration, such as `torch` for example, run

```bash
pip install -e '.[torch]'
```

Or to install just the base tango library, you can run

```bash
pip install -e .
```

### Checking your installation

Run

```bash
tango info
```

to check your installation.

### Docker image

You can build a Docker image suitable for tango projects by using [the official Dockerfile](https://github.com/allenai/tango/blob/main/Dockerfile) as a starting point for your own Dockerfile, or you can simply use one of our [prebuilt images](https://github.com/allenai/tango/pkgs/container/tango) as a base image in your Dockerfile. For example:

```Dockerfile
# Start from a prebuilt tango base image.
# You can choose the right tag from the available options here:
# https://github.com/allenai/tango/pkgs/container/tango/versions
FROM ghcr.io/allenai/tango:cuda11.3

# Install your project's additional requirements.
COPY requirements.txt .
RUN /opt/conda/bin/pip install --no-cache-dir -r requirements.txt

# Install source code.
# This instruction copies EVERYTHING in the current directory (build context),
# which may not be what you want. Consider using a ".dockerignore" file to
# exclude files and directories that you don't want on the image.
COPY . .
```

Make sure to choose the right base image for your use case depending on the version of tango you're using and the CUDA version that your host machine supports.
You can see a list of all available image tags [on GitHub](https://github.com/allenai/tango/pkgs/container/tango/versions).

<!-- end install -->

## FAQ

<!-- start faq -->

### Why is the library named Tango?

The motivation behind this library is that we can make research easier by composing it into well-defined steps.  What happens when you choreograph a number of steps together?  Well, you get a dance.  And since our [team's leader](https://nasmith.github.io/) is part of a tango band, "AI2 Tango" was an obvious choice!

### How can I debug my steps through the Tango CLI?

You can run the `tango` command through [pdb](https://docs.python.org/3/library/pdb.html). For example:

```bash
python -m pdb -m tango run config.jsonnet
```

### How is Tango different from [Metaflow](https://metaflow.org), [Airflow](https://airflow.apache.org), or [redun](https://github.com/insitro/redun)?

We've found that existing DAG execution engines like these tools are great for production workflows but not as well suited for messy, collaborative research projects
where code is changing constantly. AI2 Tango was built *specifically* for these kinds of research projects.

### How does Tango's caching mechanism work?

AI2 Tango caches the results of steps based on the `unique_id` of the step. The `unique_id` is essentially a hash of all of the inputs to the step along with:

1. the step class's fully qualified name, and
2. the step class's `VERSION` class variable (an arbitrary string).

Unlike other workflow engines like [redun](https://github.com/insitro/redun), Tango does *not* take into account the source code of the class itself (other than its fully qualified name) because we've found that using a hash of the source code bytes is way too sensitive and less transparent for users.
When you change the source code of your step in a meaningful way you can just manually change the `VERSION` class variable to indicate to Tango
that the step has been updated.

<!-- end faq -->

## Team

<!-- start team -->

**ai2-tango** is developed and maintained by the AllenNLP team, backed by [the Allen Institute for Artificial Intelligence (AI2)](https://allenai.org/).
AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.
To learn more about who specifically contributed to this codebase, see [our contributors](https://github.com/allenai/tango/graphs/contributors) page.

<!-- end team -->

## License

<!-- start license -->

**ai2-tango** is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
A full copy of the license can be found [on GitHub](https://github.com/allenai/tango/blob/main/LICENSE).

<!-- end license -->
