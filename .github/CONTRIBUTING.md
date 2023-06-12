# Contributing

Thanks for considering contributing! Please read this document to learn the various ways you can contribute to this project and how to go about doing it.

## Bug reports and feature requests

### Did you find a bug?

First, do [a quick search](https://github.com/allenai/tango/issues) to see whether your issue has already been reported.
If your issue has already been reported, please comment on the existing issue.

Otherwise, open [a new GitHub issue](https://github.com/allenai/tango/issues). Be sure to include a clear title
and description. The description should include as much relevant information as possible. The description should
explain how to reproduce the erroneous behavior as well as the behavior you expect to see. Ideally you would include a
code sample or an executable test case demonstrating the expected behavior.

### Do you have a suggestion for an enhancement or new feature?

We use GitHub issues to track feature requests. Before you create a feature request:

- Make sure you have a clear idea of the enhancement you would like. If you have a vague idea, consider discussing
  it first on a GitHub issue.
- Check the documentation to make sure your feature does not already exist.
- Do [a quick search](https://github.com/allenai/tango/issues) to see whether your feature has already been suggested.

When creating your request, please:

- Provide a clear title and description.
- Explain why the enhancement would be useful. It may be helpful to highlight the feature in other libraries.
- Include code examples to demonstrate how the enhancement would be used.

## Making a pull request

When you're ready to contribute code to address an open issue, please follow these guidelines to help us be able to review your pull request (PR) quickly.

1.  **Initial setup** (only do this once)

    <details><summary>Expand details 👇</summary><br/>

    If you haven't already done so, please [fork](https://help.github.com/en/enterprise/2.13/user/articles/fork-a-repo) this repository on GitHub.

    Then clone your fork locally with

        git clone https://github.com/USERNAME/tango.git

    or

        git clone git@github.com:USERNAME/tango.git

    At this point the local clone of your fork only knows that it came from _your_ repo, github.com/USERNAME/tango.git, but doesn't know anything the _main_ repo, [https://github.com/allenai/tango.git](https://github.com/allenai/tango). You can see this by running

        git remote -v

    which will output something like this:

        origin https://github.com/USERNAME/tango.git (fetch)
        origin https://github.com/USERNAME/tango.git (push)

    This means that your local clone can only track changes from your fork, but not from the main repo, and so you won't be able to keep your fork up-to-date with the main repo over time. Therefore you'll need to add another "remote" to your clone that points to [https://github.com/allenai/tango.git](https://github.com/allenai/tango). To do this, run the following:

        git remote add upstream https://github.com/allenai/tango.git

    Now if you do `git remote -v` again, you'll see

        origin https://github.com/USERNAME/tango.git (fetch)
        origin https://github.com/USERNAME/tango.git (push)
        upstream https://github.com/allenai/tango.git (fetch)
        upstream https://github.com/allenai/tango.git (push)

    Finally, you'll need to create a Python 3 virtual environment suitable for working on this project. There a number of tools out there that making working with virtual environments easier.
    The most direct way is with the [`venv` module](https://docs.python.org/3.8/library/venv.html) in the standard library, but if you're new to Python or you don't already have a recent Python 3 version installed on your machine,
    we recommend [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

    On Mac, for example, you can install Miniconda with [Homebrew](https://brew.sh/):

        brew install miniconda

    Then you can create and activate a new Python environment by running:

        conda create -n tango python=3.9
        conda activate tango

    Once your virtual environment is activated, you can install your local clone in "editable mode" with

        pip install -U pip setuptools wheel
        pip install -e '.[dev,all]'

    The "editable mode" comes from the `-e` argument to `pip`, and essential just creates a symbolic link from the site-packages directory of your virtual environment to the source code in your local clone. That way any changes you make will be immediately reflected in your virtual environment.

    To test your installation, just run

        tango info

    </details>

2.  **Ensure your fork is up-to-date**

    <details><summary>Expand details 👇</summary><br/>

    Once you've added an "upstream" remote pointing to [https://github.com/allenai/tango.git](https://github.com/allenai/tango), keeping your fork up-to-date is easy:

        git checkout main  # if not already on main
        git pull --rebase upstream main
        git push

    </details>

3.  **Create a new branch to work on your fix or enhancement**

    <details><summary>Expand details 👇</summary><br/>

    Committing directly to the main branch of your fork is not recommended. It will be easier to keep your fork clean if you work on a separate branch for each contribution you intend to make.

    You can create a new branch with

        # replace BRANCH with whatever name you want to give it
        git checkout -b BRANCH
        git push -u origin BRANCH

    </details>

4.  **Test your changes**

    <details><summary>Expand details 👇</summary><br/>

    Our continuous integration (CI) testing runs [a number of checks](https://github.com/allenai/tango/actions) for each pull request on [GitHub Actions](https://github.com/features/actions). You can run most of these tests locally, which is something you should do _before_ opening a PR to help speed up the review process and make it easier for us.

    First, you should run [`isort`](https://github.com/PyCQA/isort) and [`black`](https://github.com/psf/black) to make sure you code is formatted consistently.
    Many IDEs support code formatters as plugins, so you may be able to setup isort and black to run automatically everytime you save.
    For example, [`black.vim`](https://github.com/psf/black/tree/master/plugin) will give you this functionality in Vim. But both `isort` and `black` are also easy to run directly from the command line.
    Just run this from the root of your clone:

        isort .
        black .

    Our CI also uses [`ruff`](https://github.com/charliermarsh/ruff) to lint the code base and [`mypy`](http://mypy-lang.org/) for type-checking. You should run both of these next with

        ruff check .

    and

        mypy .

    We also strive to maintain high test coverage, so most contributions should include additions to [the unit tests](https://github.com/allenai/tango/tree/main/tests). These tests are run with [`pytest`](https://docs.pytest.org/en/latest/), which you can use to locally run any test modules that you've added or changed.

    For example, if you've fixed a bug in `tango/a/b.py`, you can run the tests specific to that module with

        pytest -v tests/a/b_test.py

    If your contribution involves additions to any public part of the API, we require that you write docstrings
    for each function, method, class, or module that you add.
    See the [Writing docstrings](#writing-docstrings) section below for details on the syntax.
    You should test to make sure the API documentation can build without errors by running

        make docs

    If the build fails, it's most likely due to small formatting issues. If the error message isn't clear, feel free to comment on this in your pull request.

    And finally, please update the [CHANGELOG](https://github.com/allenai/tango/blob/main/CHANGELOG.md) with notes on your contribution in the "Unreleased" section at the top.

    After all of the above checks have passed, you can now open [a new GitHub pull request](https://github.com/allenai/tango/pulls).
    Make sure you have a clear description of the problem and the solution, and include a link to relevant issues.

    We look forward to reviewing your PR!

    </details>

### Writing docstrings

We use [Sphinx](https://www.sphinx-doc.org/en/master/index.html) to build our API docs, which automatically parses all docstrings
of public classes and methods. All docstrings should adhere to the [Numpy styling convention](https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html).

## Adding a new integration

In order to add a new integration, there are several additional steps and guidelines you should follow
in addition to everything listed in [Making a pull request](#making-a-pull-request).

1. First start by creating a new submodule `tango.integrations.name_of_integration` and put all of the code for your integration in there.
2. Then you must add a module docstring to the `__init__.py` file of the submodule which imports all of the public components of the integration,
   and defines the [`__all__`](https://docs.python.org/3/tutorial/modules.html#importing-from-a-package) special variable to include all of those components.
   This ensures all of the public components will show up in the documentation.
3. Next that you should add unit tests of your code to `tests/integrations/name_of_integration/`.
4. Then add a new file `docs/source/api/integrations/name_of_integration.rst`, and include the directive:

   ```
   .. automodule:: tango.integrations.name_of_integration
      :members:
   ```

   Take a look at any of the other files in that folder to see how it should look exactly.

5. And then add `name_of_integration` to the `toctree` in `docs/source/api/integrations/index.rst`.
6. After that, add any additional requirements that your integration depends on to `requirements.txt`. Be sure to put those under the "Extra dependencies for integrations" section,
   and add the special inline comment `# needed by: name_of_integration`.
7. And finally, in the `checks` job definition in `.github/workflows/main.yml`, add a new object
   to the matrix for your integration following the other examples there.
