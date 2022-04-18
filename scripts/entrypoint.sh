#!/bin/bash

# Exit script if any commands fail.
set -e
set -o pipefail

# Check that the environment variable has been set correctly
if [ -z "$COMMIT_SHA" ]; then
  echo >&2 'error: missing COMMIT_SHA environment variable'
  exit 1
fi

# Clone and install tango.
git clone https://github.com/allenai/tango.git
cd tango
git checkout --quiet "$COMMIT_SHA"
/opt/conda/bin/pip install --no-cache-dir '.[dev,all]'

# Create directory for results.
mkdir -p /results

# Execute the arguments to this script as commands themselves, piping output into a log file.
exec "$@" 2>&1 | tee /results/out.log
