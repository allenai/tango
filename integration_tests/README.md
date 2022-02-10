# Integration tests

These are a collection of longer running end-to-end tests of various parts of the Tango library.

The easiest way to run any of these integration tests is by triggering the [**Integration tests**](https://github.com/allenai/tango/actions/workflows/integration_tests.yml)
workflow on GitHub Actions. Just select the "Run workflow" dropdown, then pick the test to run and the Beaker cluster to run it on,
and finally hit the "Run workflow" button.

Each test should have a `run.sh` file in its folder that will run the relevant tango command.
This is what the **Integration tests** workflow will call, and you can also use it to run the test manually.
