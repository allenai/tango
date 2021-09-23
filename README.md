# python-package-template

This is a template repository for Python package projects.

## Features

This template repo comes with all of the boiler plate for:

- Robust CI with GitHub Actions.
- Dependabot configuration.
- Great looking API documentation built using [Sphinx](https://www.sphinx-doc.org/en/master/) (run `make docs` to preview).
- Automatic GitHub and PyPI releases. Just follow the steps in [`RELEASE_PROCESS.md`](./RELEASE_PROCESS.md) to trigger a new release.

## Usage

1. [Create a new repository](https://github.com/allenai/python-package-template/generate) from this template with the desired name of your Python package.

    It's important that the name of the repository and the name of the Python package match exactly, otherwise you might end up with some broken links.
    So you might want to check on [PyPI](https://pypi.org/) first to see if the name is already taken.

2. Change the name of the `my_package` directory to the name of your repo / Python package.

3. Replace all mentions of `my_package` throughout this repository with the new name.

    On OS X, a quick way to find all mentions of `my_package` is:

    ```bash
    find . -type f -not -path './.git/*' -not -path ./README.md -not -path './docs/build/*' -not -path '*__pycache__*' | xargs grep 'my_package'
    ```

    There is also a one-liner to find and replace all mentions `my_package` with `actual_name_of_package`:

    ```bash
    find . -type f -not -path './.git/*' -not -path ./README.md -not -path './docs/build/*' -not -path '*__pycache__*' -exec sed -i '' -e 's/my_package/actual_name_of_package/' {} \;
    ```

3. Add repository secrets for `PYPI_USERNAME` and `PYPI_PASSWORD`.

4. Commit and push your changes, then make sure all CI checks pass.

5. Go to [readthedocs.org](https://readthedocs.org/dashboard/import/?) and import your new project.
    Then click on the "Admin" button, navigate to "Automation Rules" in the sidebar, click "Add Rule", and then enter the following fields:

    - **Description:** Publish new versions from tags
    - **Match:** Custom Match
    - **Custom match:** v[vV]
    - **Version:** Tag
    - **Action:** Activate version

    Then hit "Save".

    After your first release, the docs will automatically be publish to [your-project-name.readthedocs.io](https://your-project-name.readthedocs.io/).
