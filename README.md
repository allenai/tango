# [AI2 Tango](https://ai2-tango.readthedocs.io/)

<!-- start tagline -->

AI2 Tango replaces messy directories and spreadsheets full of file versions by organizing experiments into discrete steps that can be cached and reused throughout the lifetime of a research project.

<!-- end tagline -->

<!-- <p align="center"> -->
<p>
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
</p>

## Quick links

- [Documentation](https://ai2-tango.readthedocs.io/)
- [PyPI Package](https://pypi.org/project/ai2-tango/)
- [Contributing](https://github.com/allenai/tango/blob/main/CONTRIBUTING.md)
- [License](https://github.com/allenai/tango/blob/main/LICENSE)

## Installation

<!-- start py version -->

**ai2-tango** requires Python 3.7 or later.

<!-- end py version -->

### Installing with `pip`

<!-- start install pip -->

**ai2-tango** is available [on PyPI](https://pypi.org/project/ai2-tango/). Just run

```bash
pip install ai2-tango
```

To install with a specific integration, such as `torch` for example, run

```bash
pip install ai2-tango[torch]
```

<!-- end install pip -->

### Installing from source

<!-- start install source -->

To install **ai2-tango** from source, first clone [the repository](https://github.com/allenai/tango):

```bash
git clone https://github.com/allenai/tango.git
cd tango
```

Then run

```bash
pip install -e .
```

To install with a specific integration, such as `torch` for example, run

```bash
pip install -e .[torch]
```

<!-- end install source -->

### Checking your installation

<!-- start check install -->

Run

```bash
tango info
```

to check your installation.

<!-- end check install -->

## FAQ

### Why is the library named Tango?

The motivation behind this library is that we can make research easier by composing it into well-defined steps.  What happens when you choreograph a number of steps together?  Well, you get a dance.  And since our [team's leader](https://nasmith.github.io/) is part of a tango band, "AI2 Tango" was an obvious choice!

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
