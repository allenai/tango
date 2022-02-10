# Integration tests

These are a collection of longer running end-to-end tests of various parts of the Tango library.
At the moment we are not running these in CI since they are computationally expensive and/or require special hardware,
so they need to be run manually.

The README in each test's folder should explain where and when that particular test should be run.
Each test should have a `run.sh` file in its folder that will run the relevant tango command.
