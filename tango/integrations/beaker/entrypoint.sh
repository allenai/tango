#!/bin/bash
#
# This is the entrypoint script that the Beaker Executor uses when it runs a step
# on Beaker.
# It will work on any Docker image that has bash and conda / miniconda installed.

set -eo pipefail

# Ensure we have all the environment variables we need.
for env_var in "$GITHUB_TOKEN" "$GITHUB_REPO" "$GIT_REF"; do
    if [[ -z "$env_var" ]]; then
        echo >&2 "error: required environment variable is empty"
        exit 1
    fi
done

# Initialize conda for bash.
# See https://stackoverflow.com/a/58081608/4151392
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

echo "
[TANGO] [1/3] Installing prerequisites...
"

# Install GitHub CLI.
if ! command -v gh &> /dev/null; then
    conda install gh --channel conda-forge
fi

# Configure git to use GitHub CLI as a credential helper so that we can clone private repos.
gh auth setup-git

echo "
[TANGO] [2/3] Cloning source code from '$GITHUB_REPO'...
"

# Clone the repo and checkout the target commit.
gh repo clone "$GITHUB_REPO" src
cd src
git checkout "$GIT_REF"

echo "
[TANGO] [3/3] Reconstructing Python env...
"

if [[ -z "$VENV_NAME" ]]; then
    VENV_NAME=venv
fi
if [[ -z "$CONDA_ENV_FILE" ]]; then
    # shellcheck disable=SC2296
    CONDA_ENV_FILE="environment.yml"
fi
if [[ -z "$PIP_REQUIREMENTS_FILE" ]]; then
    # shellcheck disable=SC2296
    PIP_REQUIREMENTS_FILE="requirements.txt"
fi

if conda activate $VENV_NAME &>/dev/null; then
    echo "[TANGO] Using existing conda environment '$VENV_NAME'"
    # The virtual environment already exists. Possibly update it based on an environment file.
    if [[ -f "$CONDA_ENV_FILE" ]]; then
        echo "[TANGO] Updating environment from conda env file '$CONDA_ENV_FILE'..."
        conda env update -f "$CONDA_ENV_FILE"
    fi
else
    # The virtual environment doesn't exist yet. Create it.
    if [[ -f "$CONDA_ENV_FILE" ]]; then
        # Create from the environment file.
        echo "[TANGO] Initializing environment from conda env file '$CONDA_ENV_FILE'..."
        conda env create -n "$VENV_NAME" -f "$CONDA_ENV_FILE" 
    elif [[ -z "$PYTHON_VERSION" ]]; then
        # Create a new empty environment with the whatever the default Python version is.
        echo "[TANGO] Initializing environment with default Python version..."
        conda create -n "$VENV_NAME" pip
    else
        # Create a new empty environment with the specific Python version.
        echo "[TANGO] Initializing environment with Python $PYTHON_VERSION..."
        conda create -n "$VENV_NAME" "python=$PYTHON_VERSION" pip
    fi
    conda activate "$VENV_NAME"
fi

# Every time Beaker changes their APIs, we need to upgrade beaker-py. This happens all the
# time, so we make sure we have the latest.
# We do this when the conda environment is up, but before the requirements, so that
# requirements can request a particular beaker-py version if they want.
pip install --upgrade beaker-py

if [[ -z "$INSTALL_CMD" ]]; then
    # Check for a 'requirements.txt' and/or 'setup.py' file.
    if [[ -f 'setup.py' ]] && [[ -f "$PIP_REQUIREMENTS_FILE" ]]; then
        echo "[TANGO] Installing packages from 'setup.py' and '$PIP_REQUIREMENTS_FILE'..."
        pip install . -r "$PIP_REQUIREMENTS_FILE"
    elif [[ -f 'setup.py' ]]; then
        echo "[TANGO] Installing packages from 'setup.py'..."
        pip install .
    elif [[ -f "$PIP_REQUIREMENTS_FILE" ]]; then
        echo "[TANGO] Installing dependencies from '$PIP_REQUIREMENTS_FILE'..."
        pip install -r "$PIP_REQUIREMENTS_FILE"
    fi
else
    echo "[TANGO] Installing packages with given command: $INSTALL_CMD"
    eval "$INSTALL_CMD"
fi

PYTHONPATH="$(pwd)"
export PYTHONPATH

echo "
Environment info:
"

echo "Using $(python --version) from $(which python)"
echo "Packages:"
if which sed >/dev/null; then
    pip freeze | sed 's/^/- /'
else
    pip freeze
fi

echo "
[TANGO] Setup complete ✓
"

# Execute the arguments to this script as commands themselves.
exec "$@"
