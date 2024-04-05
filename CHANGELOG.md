# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Fixed

- Fixed a bunch of dependencies
- Upgraded to new version of wandb

## [v1.3.2](https://github.com/allenai/tango/releases/tag/v1.3.2) - 2023-10-27

### Fixed

- Fix issues with gcloud auth in beaker executor.

## [v1.3.1](https://github.com/allenai/tango/releases/tag/v1.3.1) - 2023-10-25

### Fixed

- Minor bugs in the `GSWorkspace()`.

### Changed

- Added CLI-style execution functions for experiments defined in Python.
- Added `display()` to `ExecutorOutput` for producing a table that summarizes the run.

## [v1.3.0](https://github.com/allenai/tango/releases/tag/v1.3.0) - 2023-10-13

### Added
 - Added the `Workspace.remove_step()` method to safely remove steps.
- The `GSWorkspace()` can now be initialized with google cloud bucket subfolders.

### Changed

- The `BeakerExecutor` now uses the HEAD commit at the time the executor is instantiated to executor a step instead of the HEAD commit at the time the step is run.

### Fixed

- Removed unnecessary code coverage dev requirements.
- Fixed issue where new version of torch caused no LR schedulers to be registered.
- Updated pinned versions of jax, jaxlib, and flax.

## [v1.2.1](https://github.com/allenai/tango/releases/tag/v1.2.1) - 2023-04-06

### Added

- Added the following workspace methods to support the Tango viz UI: `Workspace.search_registered_runs()`, `Workspace.search_step_info()`, `Workspace.num_registered_runs()`, and `Workspace.num_steps()`.

### Fixed

- Fixes a bug where `FromParams` would fail to parse when an object takes a `Step` argument directly.
- Changed a name so we don't override the built-in name `set`.
- Fixed a bug that would cause O(n^2) memory consumption in dense step graphs.


## [v1.2.0](https://github.com/allenai/tango/releases/tag/v1.2.0) - 2023-02-10

### Added

- You can now add arguments to steps without invalidating the cache. See `Step.SKIP_DEFAULT_ARGUMENTS`.
- Fixed integration status messages in `tango info` command.
- Added abstractions for `RemoteClient`, `RemoteStepCache`, and `RemoteWorkspace`.
- Added a GS integration that comes with `GSWorkspace`, a remote `Workspace` implementation that uses google cloud storage.
- You can now bind functional steps to the underlying `Step` instance with `@step(bind=True)`, meaning the first argument to the function will be a `Step`.
- Added `ShellStep` for running arbitrary shell commands.
- Added `@make_registrable` decorator to make arbitrary functions registrable, to make it easier to refer to them in tango configurations.

### Fixed

- Jsonnet parsing is now much faster and works on Windows.
- Warnings about locks are now reliably printed every 30 seconds
- We now make sure Beaker jobs have the latest version of beaker-py, so that we're compatible with the latest API changes.
- Stopping early now works when the metric doesn't change at all.
- Fixed bug with `FromParams` which didn't handle variable length tuples correctly.

### Changed

- The default log level for Tango is now `warning`.
- You can specify multiple steps with `-s` from the `tango run` command.


## [v1.1.0](https://github.com/allenai/tango/releases/tag/v1.1.0) - 2022-12-01

### Added

- Added `gpu_type` field to `StepResources`. The `BeakerExecutor` can use this to determine which clusters to a submit a step to.
- Added `machine` field to `StepResources`. You can set this to "local" when using the `BeakerExecutor` to force it to run the step locally.
- Added `--ext-var` argument to `tango run` for setting JSONNET external variables
  when loading the experiment config.
- Added `@step()` decorator to create `Step` classes from functions.
- Added the `transformers::with_soft_prompt` integration, to make soft-prompted prefix transformers easy.

### Removed

- Removed PyTorch Lightning integration.
- Removed `tango server` command and `--serve/--no-serve` option for `tango run`.
- Removed `source_release.py`, which was checked in by accident.

### Fixed

- Fixed issue where Executor `parallelism` option in a Tango settings file would be ignored.
- Fixed a bug where the unique ID of a step that depends on a key-value of the result of another step could change if the name of the other step changes.
- Fixed a bug where importing certain libraries (like torchmetrics) would mess with our exception handling because they set `sys.excepthook` for some reason. Now we always reset `sys.excepthook` after importing.
- The type hints for the flax trainer suggested that the training split is optional when in fact it's mandatory.
- Made `BeakerWorkspace` / `BeakerStepLock` more robust when a job is preempted.
- Minor performance improvements for the Beaker executor and workspace.
- Fixed bug with `step_extra_dependencies` where uncacheable dependencies wouldn't be run.


## [v1.0.2](https://github.com/allenai/tango/releases/tag/v1.0.2) - 2022-11-14

### Changed

- `BeakerScheduler` can now return a list of clusters.

## [v1.0.1](https://github.com/allenai/tango/releases/tag/v1.0.1) - 2022-10-20

### Fixed

- `LightningTrainStep` now can take a `Lazy` model object which results in a gauranteed deterministic hash.
- Fixed issue where remote `Workspace` implementations like `WandbWorkspace` and `BeakerWorkspace` would use the same local cache regardless of the W&B / Beaker workspace
  being used.
- Fixed bug with `TorchEvalStep` when constructing callbacks.
- Fixed some import error issues caused when an integration is not installed.
- Fix incorrect reporting of final results in `MulticoreExecutor`.

### Changed

- Wandb step cache retries api call in case of timeout
- `beaker-py >= 1.11` required.

## [v1.0.0](https://github.com/allenai/tango/releases/tag/v1.0.0) - 2022-10-05

### Added

- Added `step_extra_dependencies` input field to `Step` class that can be used to force a dependency on another step even if the current step doesn't directly depend on the output of the other step. See [#418](https://github.com/allenai/tango/issues/418) for more context.

### Changed

- `beaker-py >= 1.10` required.

### Fixed

- Long log lines will be soft-wrapped to ensure that links are clickable.
- Fixed a bug where some workspaces could be left in a bad state if a step's `Format` failed to serialize the step's result in `Workspace.step_finished()`.
- Sometimes functions and methods end up as arguments to steps, which means we have to hash them. Instead of taking
  a hash of the function, we now take a hash of the function's module and name.
- Fixed a bug with the Beaker executor where it would hang at the end of a run if a step failed that is a dependency of another step.
- Fixed tests to work with new version of transformers.
- Fixed `Executor.execute_sub_graph_for_step()` to be able to run the step's dependencies in parallel.


## [v0.14.0](https://github.com/allenai/tango/releases/tag/v0.14.0) - 2022-09-20

### Added

- Adds a function to modify a Hugging Face transformer with IA3 adaptors
- Added a `BeakerScheduler` registrable class, specified as the argument `scheduler` to `BeakerExecutor`, which controls the resources assigned to steps ran on Beaker.
  Users can implement their own `BeakerScheduler` subclasses to customize the resource assignment behavior.

### Changed

- In the `tango run` command, `--no-server` is now the default. Use `--server` to start the server.

### Fixed

- Made `BeakerExecutor` more robust to connection, timeout, SSL, and other recoverable HTTP errors.
- Made the `BeakerStepLock` more robust, and as a result `BeakerWorkspace` is more
  robust and should require less manual intervention for locks in a bad state.
- Fixed a bug with the internal scheduling logic of the `BeakerExecutor` which
  could delay submitting some steps in parallel.
- Fixed a bug where creating a `StepInfo` object from params might result in unnecessary imports.
- Fixed a bug where canceling the Beaker executor might not work properly.
- Fixed a bug where the trainer trains too much when `train_epochs` is set and you're using gradient accumulation.
- Fixed a bug where included modules might not be found when using multiprocessing when they're not on `sys.path` / `PYTHONPATH`.
- Fixed how the results of uncacheable steps are displayed by `tango run`.
- Beaker executor won't run duplicate cacheable steps at the same time.

## [v0.13.0](https://github.com/allenai/tango/releases/tag/v0.13.0) - 2022-09-07

### Added

- You can now reference into a particular index of the result of another step in a config. For example: `{type: "ref", ref: "some_previous_step", key: 0}`.
  The key field can be an integer if the result of the referenced step is a list or tuple, or a string if the result of the referenced step is a dictionary.
- Added `priority` parameter to Beaker executor for setting the default task priority for Beaker jobs.
- Added `Workspace.step_result()` method for getting a step's result from the latest
  run.
- `tango run` will now display a URL to the logs for failed steps when you use the `BeakerExecutor`.

### Changed

- The `TorchTrainStep` now enables monitoring arbitrary model outputs during training. `TorchTrainEngine.forward_train` now returns a tuple `loss, model_outputs` for each micro batch and the list of model outputs for all micro batches in a batch is passed to the `TrainCallback.log_batch` and `TrainCallback.post_batch`.
- Tango will now automatically search Python modules in the current working directory
  for registered classes so that you don't always need to use the `--include-package` setting.
- The minimum supported Python version is now 3.8.
- Added support for PyTorch Lightning 1.7.x
- The Beaker Executor will no-longer live-stream logs from Beaker jobs, but logs will be viewable on Beaker and more readable.
- Only the Beaker executor requires a clean working directory

### Fixed

- Fixed a bug that did not allow a wandb artifact's type to be set from a step's metadata dictionary. 
- Fixed a bug with how the Beaker executor streams log lines from Beaker which sometimes resulted in messages missing some starting characters, and tqdm lines being duplicated.
- Fixed a bug in the Beaker workspace where the lock dataset wouldn't be removed if the step
  was found to be in an invalid state.
- Improved cluster choice logic in `BeakerExecutor` to ensure greater diversity of clusters when submitting many steps at once.
- Fixed bug where sub-processes of the multicore executor would use the wrong executor if `executor` was defined in a `tango.yml` file.
- Deterministic hashes for numpy and torch tensors were not deterministic. Now they are.


## [v0.12.0](https://github.com/allenai/tango/releases/tag/v0.12.0) - 2022-08-23

### Added

- **Step resources:**
  - Added a `step_resources` parameter to the `Step` class which should be used to describe the computational resources required to run a step.
    `Executor` implementations can use this information. For example, if your step needs 2 GPUs, you should set
    `step_resources=StepResources(gpu_count=2)` (`"step_resources": {"gpu_count": 2}` in the configuration language).
  - Added a `Step.resources()` property method. By default this returns the value specified by the `step_resources` parameter.
    If your step implementation always requires the same resources, you can just override this method so you don't have to provide
    the `step_resources` parameter.
- **Step execution:**
  - Added an `executor` field to the `tango.yml` settings. You can use this to define the executor you want to use by default.
  - Added a Beaker `Executor` to the Beaker integration, registered as an `Executor` with the name "beaker".
    To use this executor, add these lines to your `tango.yml` file:
    ```yaml
    executor:
      type: beaker
      beaker_workspace: ai2/my-workspace
      clusters:
        - ai2/general-cirrascale
    ```
    See the docs for the `BeakerExecutor` for more information on the input parameters.
- **Step class:**
  - Added a metadata field to the step class API. This can be set through the class
    variable `METADATA` or through the constructor argument `step_metadata`.
- **Weights & Biases integration:**
  - You can now change the artifact kind for step result artifacts by adding a field
    called "artifact_kind" to a step's metadata.
    For models, setting "artifact_kind" to "model" will add the corresponding artifact to W&B's new model zoo.

### Changed

- **CLI:**
  - The `tango run` command will throw an error if you have uncommitted changes in your repository, unless
    you use the `--allow-dirty` flag.
  - The `tango run` command will use the lightweight base executor (single process) by default.
    To use the multi-process executor, set `-j/--parallelism` to 1 or higher or -1 to use all available CPU cores.

### Fixed

- Fixed bug where `StepInfo` environment and platform metadata could be out-of-date if a step is run again due to failure.
- Fixed a bug where an unfortunate combination of early stopping and decreasing model performance could result in a crash in the torch trainer.

## [v0.11.0](https://github.com/allenai/tango/releases/tag/v0.11.0) - 2022-08-04

### Added

- Added a [Flax](https://flax.readthedocs.io/en/latest/) integration along with an example config.

## [v0.10.1](https://github.com/allenai/tango/releases/tag/v0.10.1) - 2022-07-26

### Fixed

- Fixed issue where the StepInfo config argument could be parsed into a Step. 
- Restored capability to run tests out-of-tree.

## [v0.10.0](https://github.com/allenai/tango/releases/tag/v0.10.0) - 2022-07-07

### Changed

- Renamed `workspace` parameter of `BeakerWorkspace` class to `beaker_workspace`.
- `Executor` class is now a `Registrable` base class. `MulticoreExecutor` is registered as "multicore".

### Removed

- Removed `StepExecutionMetadata`. Its fields have been absorbed into `StepInfo`.

### Fixed

- Improved `Step.ensure_result()` such that the step's result doesn't have to be read from the cache.
- Fixed an issue with the output from `MulticoreExecutor` such that it's now consistent with the default `Executor` for steps that were found in the cache.
- One of our error messages referred to a configuration file that no longer exists.
- Improved performance of `BeakerWorkspace`.

### Added

- Added the ability to train straight `Model` instead of just `Lazy[Model]`


## [v0.9.1](https://github.com/allenai/tango/releases/tag/v0.9.1) - 2022-06-24

### Fixed

- Fixed non-deterministic behavior in `TorchTrainStep`.
- Fixed bug in `BeakerWorkspace` where `.step_info(step)` would raise a `KeyError` if the step hasn't been registered as part of a run yet.
- Fixed a bug in `BeakerWorkspace` where it would send too many requests to the beaker service.
- Fixed a bug where `WandbWorkspace.step_finished()` or `.step_failed()` would crash if called
  from a different process than `.step_starting()`.
- Fixed a bug in `WandbWorkspace.step_finished()` which led to a `RuntimeError` sometimes while
  caching the result of a step.


## [v0.9.0](https://github.com/allenai/tango/releases/tag/v0.9.0) - 2022-06-01

### Added

- Added a [Beaker](https://beaker.org) integration that comes with `BeakerWorkspace`, a remote `Workspace` implementation that uses Beaker Datasets under the hood.
- Added a `datasets::dataset_remix` step that provides the split remixing functionality of `tango.steps.datasest_remix.DatasetRemixStep` now for Huggingface `DatasetDict`.
- Added a config and code example of Registrable to the First Step docs with edits for clarity.

### Changed

- If you try to import something from a tango integration that is not fully installed due to missing dependencies, an `IntegrationMissingError` will be raised
instead of `ModuleNotFound`.
- You can now set `-j 0` in `tango run` to disable multicore execution altogether.

### Fixed

- Improved how steps and workspaces handle race conditions when different processes are competing to execute the same step. This would result in a `RuntimeError` before with most workspaces, but now it's handled gracefully.
- Fixed bug which caused GradScaler state to not be saved and loaded with checkpoints. 

## [v0.8.0](https://github.com/allenai/tango/releases/tag/v0.8.0) - 2022-05-19

### Added

- Added a Weights & Baises remote `Workspace` implementation: `WandbWorkspace`, registered as "wandb".
  This can be instantiated from a workspace URL in the form "wandb://entity/project".
- Added a method `Workspace.step_result_for_run` which gives the result of a step given the run name and step name within that run.
- Added property `Workspace.url`, which returns a URL for the workspace that can be used to instantiate the exact same workspace using `Workspace.from_url()`. Subclasses must implement this.

### Changed

- `StepInfo` start and end times will be always be in UTC now.
- `WandbTrainCallback` now logs system metrics from each worker process in distributed training.
- `StepCache.__contains__()` and `StepCache.__getitem__()` now take accept either a `Step` or `StepInfo` as an argument (`Union[Step, StepInfo]`).
- Refactored `tango.step_graph.StepGraph` to allow initialization from a `Dict[str, Step]`.
- `Executor.execute_step_graph()` now attempts to execute all steps and summarizes success/failures.

### Fixed

- Fixed bug with `LocalWorkspace.from_parsed_url()` ([#278](https://github.com/allenai/tango/issues/278)).
- Deprecation warnings will now be logged from `tango` CLI.
- Fixed the text format in the case of serializing an iterator of string.
- Added missing default value of `None` to `TangoGlobalSettings.find_or_default()`.
- Mypy has become incompatible with transformers and datasets, so we have to disable the checks in some places.
- The `VERSION` member of step arguments that were wrapped in `Lazy` were not respected. Now they are.


## [v0.7.0](https://github.com/allenai/tango/releases/tag/v0.7.0) - 2022-04-19

### Added

- Added the "-n/--name" option to `tango run`. This option allows the user to give the run an arbitrary name.
- Added a convenience property `.workspace` to `Step` class that can be called from a step's `.run()` method to get the current `Workspace` being used.
- Gave `FromParams` objects (which includes all `Registrable` objects) the ability to version themselves.
- Added CLI option to run a single step in a config using `--step-name` or `-s`.
- Added a `MultiCoreExecutor` that executes steps in parallel.
- Added an `ExecutorOutput` dataclass that is returned by `Executor.execute_step_graph()`.
- `StepGraph` now prints itself in a readable way.
- Tango now automatically detects when it's running under a debugger, and disables multicore support accordingly. Many debuggers can't properly follow sub-processes, so this is a convenience for people who love debuggers.
- Added more models to the stuff we can import from the transformers library.
- Added new example for finetuning text-to-text models.

### Changed

- Renamed `click_logger` to `cli_logger`, and we now use [rich](https://github.com/Textualize/rich)'s logging `Handler` as the default handler, which means prettier output, better tracebacks, and you can use rich's markup syntax with the `cli_logger` to easily add style to text.
- Refactored `tango.step_graph.StepGraph` to allow initialization from a `Dict[str, Step]`.
- `Executor.execute_step_graph()` now attempts to execute all steps and summarizes success/failures.
- Upgraded PyTorch version in `tango` Docker image to latest `v1.11.0+cu113`.
- `RunGeneration` now allows model object as input.

### Fixed

- Fixed bug that mistakenly disallowed fully-qualified names containing `"_"` (underscores) in the config.
- Fixed bug where `TorchTrainStep` working directory would be left in an unrecoverable state if training failed after saving the final model weights.
- Fixed bug in `FromParams` where `**kwargs` might be passed down to the constructors of arguments.
- Fixed bug in the way dependencies are tracked between steps.
- Fixed bug that caused `MulticoreExecutor` to hang in case of a failing step that was required recursively (not directly) downstream.
- Fixed bug in the way dependencies are tracked between steps
- Compatibility with PyTorch Lightning 1.6


## [v0.6.0](https://github.com/allenai/tango/releases/tag/v0.6.0) - 2022-02-25

### Added

- New example that finetunes a pre-trained ResNet model on the Cats & Dogs dataset.
- Added a '@requires_gpus' decorator for marking tests as needing GPUs. Tests marked with this will be run in the "GPU Tests" workflow
  on dual k80 GPUs via Beaker.
- Added the "-w/--workspace" option to `tango run` and `tango server` commands. This option takes a path or URL, and instantiates the workspace from the URL using the newly added `Workspace.from_url()` method.
- Added the "workspace" field to `TangoGlobalSettings`.
- Added the "environment" field to `TangoGlobalSettings` for setting environment variables each
  time `tango` is run.
- Added a utility function to get a `StepGraph` directly from a file.
- Added `tango.settings` module and `tango settings` group of commands.
- A format for storing sequences as `SqliteSparseSequence`
- A way to massage kwargs before they determine the unique ID of a `Step`

### Changed

- `local_workspace.ExecutorMetadata` renamed to `StepExecutionMetadata` and now saved as `execution-metadata.json`.
- `tango run` without the option "-w/--workspace" or "-d/--workspace-dir" will now use a `MemoryWorkspace` instead of a `LocalWorkspace` in a temp directory, unless you've specified
  a default workspace in a `TangoGlobalSettings` file.
- Moved `tango.workspace.MemoryWorkspace` and `tango.local_workspace.LocalWorkspace` to `tango.workspaces.*`.
- Moved `tango.step_cache.MemoryStepCache` and `tango.step_cache.LocalStepCache` to `tango.step_caches.*`.
- Deprecated the `-d/--workspace-dir` command-line option. Please use `-w/--workspace` instead.

### Fixed

- Fixed a small bug `LocalWorkspace` would fail to capture the conda environment in our Docker image.
- Fixed activation of `FILE_FRIENDLY_LOGGING` when set from the corresponding environment variable.
- Fixed setting log level via the environment variable `TANGO_LOG_LEVEL`.
- Use relative paths within the `work_dir` for symbolic links to the latest and the best checkpoints in `TorchTrainStep`.
- Fixed some scenarios where Tango can hang after finishing all steps.
- `distributed_port` and `log_every` parameters won't factor into `TorchTrainStep`'s unique ID.
- `MappedSequence` now works with slicing.
- `MappedSequence` now works with Huggingface `Dataset`.
- Uncacheable steps are now visible in Tango UI.
- Fixed bug in `Registrable.list_available()` where an error might be raised if the default implementation hadn't been explicitly imported.
- Fixed issue where having a default argument to the `run()` method wasn't getting applied to the step's unique ID.


## [v0.5.0](https://github.com/allenai/tango/releases/tag/v0.5.0) - 2022-02-09

### Added

- Added `TrainingEngine` abstraction to torch integration.
- Added [FairScale](https://fairscale.readthedocs.io/en/latest/) with a `FairScaleTrainingEngine`
  that leverages FairScale's `FullyShardedDataParallel`. This is meant to be used within the `TorchTrainStep`.
- All PyTorch components (such as learning rate schedulers, optimizers, data collators, etc) from the
  transformers library and now registered under the corresponding class in the torch integration.
  For example, transformers `Adafactor` optimizer is registered as an `Optimizer` under the name
  "transformers::Adafactor". More details can be found in the documentation for the transformers integration.

### Changed

- Various changes to the parameters othe `TorchTrainStep` due to the introduction of the `TrainingEngine` class.
- Params logged as `DEBUG` level instead of `INFO` to reduce noise in logs.
- The waiting message for `FileLock` is now clear about which file it's waiting for.
- Added an easier way to get the default Tango global config
- Most methods to `TorchTrainCallback` also take an `epoch` parameter now.
- `WandbTrainCallback` now logs peak GPU memory occupied by PyTorch tensors per worker. This is useful because W&B's system metrics only display the total GPU memory reserved by PyTorch, which is always higher than the actual amount of GPU memory occupied by tensors. So these new metrics give a more accurate view into how much memory your training job is actually using.
- Plain old Python functions can now be used in `Lazy` objects.
- `LocalWorkspace` now creates a symlink to the outputs of the latest run.
- Tango is now better at guessing when a step has died and should be re-run.
- Tango is now more lenient about registering the same class under the same name twice.
- When you use `dict` instead of `Dict` in your type annotations, you now get a legible error message. Same for `List`, `Tuple`, and `Set`.

### Fixed

- Fixed a bug in `Registrable` and `FromParams` where registered function constructors would not properly construct
  arguments that were classes.
- Fixed a bug in `FromParams` that would cause a crash when an argument to the constructor had the name `params`.
- Made `FromParams` more efficient by only trying to parse the params as a `Step` when it looks like it actually could be a step.
- Fixed bug where `Executor` would crash if `git` command could not be found.
- Fixed bug where validation settings were not interpreted the right way by the torch trainer.
- When you register the same name twice using `Registrable`, you get an error message. That error message now contains the correct class name.


## [v0.4.0](https://github.com/allenai/tango/releases/tag/v0.4.0) - 2022-01-27

### Changed

- Default log level is `WARNING` instead of `ERROR`.
- The web UI now renders the step graph left-to-right.
- The web UI now shows runs by date, with the most recent run at the top.
- The web UI now shows steps in a color-coded way.
- The `tango run` command now prints user-friendly paths if possible.
- The `--include-package` flag now also accepts paths instead of module names.
- `tango.common.sqlite_sparse_sequence.SqliteSparseSequence` now lives at `tango.common.sequences.SqliteSparseSequence`.

### Fixed

- Ensure tqdm log lines always make it into the log file `out.log` even when log level is `WARNING` or `ERROR`.
- Numerous parts of Tango now have documentation when they didn't before.


## [v0.4.0rc5](https://github.com/allenai/tango/releases/tag/v0.4.0rc5) - 2022-01-19

### Added

- Added `TorchEvalStep` to torch integration, registered as "torch::eval".

### Changed

- Renamed `aggregate_val_metric` to `auto_aggregate_val_metric` in `TorchTrainStep`.
- `devices` parameter to `TorchTrainStep` replaced with `device_count: int`.
- Run name printed at the end of a run so it's easier to find.
- Type information added to package data. See [PEP 561](https://www.python.org/dev/peps/pep-0561) for more information.
- A new integration, `transformers`, with two new steps for running seq2seq models.
- Added `logging_tqdm`, if you don't want a progress bar, but you still want to see progress in the logs.
- Added `threaded_generator()`, for wrapping generators so that they run in a separate thread from the generator's consumer.
- Added a new example for evaluating the T0 model on XSum, a summarization task.
- Added `MappedSequence` for functionally wrapping sequences.
- Added `TextFormat`, in case you want to store the output of your steps in raw text instead of JSON.
- Steps can now list arguments in `SKIP_ID_ARGUMENTS` to indicate that the argument should not affect a step's
  unique id. This is useful for arguments that affect the execution of a step, but not the output.
- `Step` now implements `__str__`, so steps look pretty in the debugger.
- Added `DatasetCombineStep`, a step that combines multiple datasets into one.
- Added `common.logging.initialize_worker_logging()` function for configuring logging from worker processes/threads.
- Logs from `tango run ...` will be written to a file called `out.log` in the run directory.

### Fixed

- Fixed torch `StopEarlyCallback` state not being recovered properly on restarts.
- Fixed file friendly logging by removing special styling characters.
- Ensured exceptions captured in logs.
- `LocalWorkspace` now works properly with uncacheable steps.
- When a Tango run got killed hard, with `kill -9`, or because the machine lost power, `LocalWorkspace` would
  sometimes keep a step marked as "running", preventing further executions. This still happens sometimes, but it
  is now much less likely (and Tango gives you instructions for how to fix it).
- To make all this happen, `LocalWorkspace` now saves step info in a Sqlite database. Unfortunately that means that
  the workspace format changes and existing workspace directories won't work properly with it.
- Fixed premature cleanup of temporary directories when using `MemoryWorkspace`


## [v0.4.0rc4](https://github.com/allenai/tango/releases/tag/v0.4.0rc4) - 2021-12-20

### Fixed

- Fixed a bug where `StepInfo` fails to deserialize when `error` is an exception that can't be pickled.


## [v0.4.0rc3](https://github.com/allenai/tango/releases/tag/v0.4.0rc3) - 2021-12-15

### Added

- Added `DatasetsFormat` format and `LoadStreamingDataset` step to `datasets` integration.
- `SqliteDictFormat` for datasets.
- Added `pre_epoch()` and `post_epoch()` callback methods to PyTorch `TrainCallback`.

### Changed

- `LoadDataset` step from `datasets` integration is now cacheable, using the `DatasetsFormat` format by default.
  But this only works with non-streaming datasets. For streaming datasets, you should use the `LoadStreamingDataset` step instead.

### Fixed

- Fixed bug where `KeyboardInterrupt` exceptions were not handled properly by steps and workspaces.
- `WandbTrainCallback` now will use part of the step's unique ID as the name for the W&B run by default, to make
  it easier to indentify which tango step corresponds to each run in W&B.
- `WandbTrainCallback` will save the entire `TrainConfig` object to the W&B config.


## [v0.4.0rc2](https://github.com/allenai/tango/releases/tag/v0.4.0rc2) - 2021-12-13

### Added

- Sample experiment configurations that prove Euler's identity

### Changed

- Loosened `Click` dependency to include v7.0.
- Loosened `datasets` dependency.
- Tightened `petname` dependency to exclude next major release for safety.

### Fixed

- `Workspace`, `MemoryWorkspace`, and `LocalWorkspace` can now be imported directly from the `tango`
  base module.
- Uncacheable leaf steps would never get executed. This is now fixed.
- We were treating failed steps as if they were completed by accident.
- The visualization had a problem with showing steps that never executed because a dependency failed.
- Fixed a bug where `Lazy` inputs to a `Step` would fail to resolve arguments that come from the result
  of another step.
- Fixed a bug in `TorchTrainStep` where some arguments for distributed training (`devices`, `distributed_port`) weren't being set properly.


## [v0.4.0rc1](https://github.com/allenai/tango/releases/tag/v0.4.0rc1) - 2021-11-30

### Added

- Introduced the concept of the `Workspace`, with `LocalWorkspace` and `MemoryWorkspace` as initial implementations.
- Added a stub of a webserver that will be able to visualize runs as they happen.
- Added separate classes for `LightningTrainingTypePlugin`, `LightningPrecisionPlugin`, `LightningClusterEnvironmentPlugin`, `LightningCheckpointPlugin` for compatibility with `pytorch-lightning>=1.5.0`.
- Added a visualization of workspaces that can show step graphs while they're executing.

### Removed

- Removed old `LightningPlugin` class
- Removed requirement of the `overrides` package

### Changed

- Made it possible to construct a step graph out of `Step` objects, instead of constructing it out of `StepStub` objects.
- Removed dataset fingerprinting code, since we can now use `Step` to make sure things are cached.
- Made steps deterministic by default.
- Brought back `MemoryStepCache`, so we can run steps without configuring anything.
- W&B `torch::TrainCallback` logs with `step=step+1` now so that training curves in the W&B dashboard
  match up with checkpoints saved locally and are easier to read (e.g. step 10000 instead of 9999).
- `filelock >= 3.4` required, parameter `poll_intervall`  to `tango.common.file_lock.FileLock.acquire` renamed
  to `poll_interval`.

### Fixed

- Fixed bug in `FromParams` where a parameter to a `FromParams` class may not be instantiated correctly
  if it's a class with a generic type parameter.

## [v0.3.6](https://github.com/allenai/tango/releases/tag/v0.3.6) - 2021-11-12

### Added

- Added a `.log_batch()` method on `torch::TrainCallback` which is given the average loss across
  distributed workers, but only called every `log_every` steps.

### Removed

- Removed `.pre_log_batch()` method on `torch::TrainCallback`.

### Fixed

- Fixed typo in parameter name `remove_stale_checkpoints` in `TorchTrainStep` (previously was `remove_state_checkpoints`).
- Fixed bug in `FromParams` that would cause failures when `from __future__ import annotations`
  was used with Python older than 3.10. See [PEP 563](https://www.python.org/dev/peps/pep-0563/)
  for details.

## [v0.3.5](https://github.com/allenai/tango/releases/tag/v0.3.5) - 2021-11-05

### Fixed

- Fixed a bug in `FromParams` where the "type" parameter was ignored in some cases
  where the `Registrable` base class did not directly inherit from `Registrable`.

## [v0.3.4](https://github.com/allenai/tango/releases/tag/v0.3.4) - 2021-11-04

### Added

- Added `StopEarlyCallback`, a `torch::TrainCallback` for early stopping.
- Added parameter `remove_stale_checkpoints` to `TorchTrainStep`.

### Changed

- Minor changes to `torch::TrainCallback` interface.
- Weights & Biases `torch::TrainCallback` now logs best validation metric score.

## [v0.3.3](https://github.com/allenai/tango/releases/tag/v0.3.3) - 2021-11-04

### Added

- Added support for PEP 604 in `FromParams`, i.e. writing union types as "X | Y" instead of "Union[X, Y]".
- [internals] Added a spot for miscellaneous end-to-end integration tests (not to be confused with "tests of integrations") in `tests/end_to_end/`.
- [internals] Core tests now run on all officially supported Python versions.

### Fixed

- Fixed a bug in `FromParams` where non-`FromParams` class parameters were not instantiated
  properly (or at all).
- Fixed a bug in `FromParams` where kwargs were not passed on from a wrapper class to the wrapped class.
- Fixed small bug where some errors from git would be printed when executor metadata is created
  outside of a git repository.

## [v0.3.2](https://github.com/allenai/tango/releases/tag/v0.3.2) - 2021-11-01

### Fixed

- Fixed a bug with `FromParams` that caused `.from_params()` to fail when the params contained
  an object that was already instantiated.
- tango command no longer installs a SIGTERM handler, which fixes some bugs with integrations that use multiprocessing.

## [v0.3.1](https://github.com/allenai/tango/releases/tag/v0.3.1) - 2021-10-29

### Changed
- Updated the `LightningTrainStep` to optionally take in a `LightningDataModule` as input.

## [v0.3.0](https://github.com/allenai/tango/releases/tag/v0.3.0) - 2021-10-28

### Added

- Added `IterableDatasetDict`, a version of `DatasetDict` for streaming-like datasets.
- Added a [PyTorch Lightning](https://www.pytorchlightning.ai) integration with `LightningTrainStep`.

### Fixed

- Fixed bug with `FromParams` and `Lazy` where extra arguments would sometimes be passed down through
  to a `Lazy` class when they shouldn't.

## [v0.2.4](https://github.com/allenai/tango/releases/tag/v0.2.4) - 2021-10-22

### Added

- Added support for [torch 1.10.0](https://github.com/pytorch/pytorch/releases).

### Changed

- `--file-friendly-logging` flag is now an option to the main `tango` command, so needs
  to be passed before `run`, e.g. `tango --file-friendly-logging run ...`.

### Fixed

- Fixed bug with `Step.from_params`.
- Ensure logging is initialized is spawn processes during distributed training with `TorchTrainStep`.

## [v0.2.3](https://github.com/allenai/tango/releases/tag/v0.2.3) - 2021-10-21

### Added

- Added support for global settings file, `tango.yml`.
- Added 'include_package' (array of string) param to config spec.
- Added a custom error `StopEarly` that a `TrainCallback` can raise within the `TorchTrainStep`
  to stop training early without crashing.
- Added step config, tango command, and tango version to executor metadata.
- Executor now also saves pip dependencies and conda environment files to the run directory
  for each step.

### Fixed

- Ensured `**kwargs` arguments are logged in `FromParams`.

## [v0.2.2](https://github.com/allenai/tango/releases/tag/v0.2.2) - 2021-10-19

### Added

- Added new steps to `datasets` integration: `ConcatenateDatasets` ("datasets::concatenate") and `InterleaveDatasets` (datasets::interleave).
- Added `__contains__` and `__iter__` methods on `DatasetDict` so that it is now a `Mapping` class.
- Added `tango info` command that - among other things - displays which integrations are installed.

## [v0.2.1](https://github.com/allenai/tango/releases/tag/v0.2.1) - 2021-10-18

### Added

- Added `convert_to_tango_dataset_dict()` function in the `datasets` integration.
  It's important for step caching purposes to use this to convert a HF `DatasetDict`
  to a native Tango `DatasetDict` when that `DatasetDict` is part of the input to another
  step. Otherwise the HF `DatasetDict` will have to be pickled to determine its hash.

### Changed

- `Format.checksum()` is now an abstract method. Subclasses should only compute checksum
  on the serialized artifact and nothing else in the directory.
- [internals] Changed the relationship between `Executor`, `StepCache`, and `Step.`
  `Executor` now owns the `StepCache`, and `Step` never interacts with `StepCache` directly.

## [v0.2.0](https://github.com/allenai/tango/releases/tag/v0.2.0) - 2021-10-15

### Added

- Added a [Weights & Biases](https://wandb.ai) integration with a training callback ("wandb::log")
  for `TorchTrainStep` ("torch::train") that logs training and validation metrics to W&B.

### Fixed

- Fixed `Format.checksum()` when there is a symlink to a directory in the cache folder.

## [v0.1.3](https://github.com/allenai/tango/releases/tag/v0.1.3) - 2021-10-15

### Added

- Added the ability to track a metric other than "loss" for validation in `TorchTrainStep` ("torch::train").

### Fixed

- Final model returned from `TorchTrainStep` ("torch::train") will have best weights loaded.
- Checkpoints are saved from `TorchTrainStep` ("torch::train") even when there is no validation loop.
- Fixed `TorchTrainStep` ("torch::train") when `validation_split` is `None`.
- Fixed distributed training with `TorchTrainStep` ("torch::train") on GPU devices.

## [v0.1.2](https://github.com/allenai/tango/releases/tag/v0.1.2) - 2021-10-13

### Added

- Added support for YAML configuration files.

## [v0.1.1](https://github.com/allenai/tango/releases/tag/v0.1.1) - 2021-10-12

### Added

- `TorchTrainStep` now displays a progress bar while saving a checkpoint to file.
- The default executor now saves a "executor-metadata.json" file to the directory for each step.

### Changed

- Renamed `DirectoryStepCache` to `LocalStepCache` (registered as "local").
- `LocalStepCache` saves metadata to `cache-metadata.json` instead of `metadata.json`.

### Fixed

- Fixed bug with `TorchTrainStep` during distributed training.
- `FromParams` will automatically convert strings into `Path` types now when the annotation
  is `Path`.

## [v0.1.0](https://github.com/allenai/tango/releases/tag/v0.1.0) - 2021-10-11

### Added

- Added `StepGraph` and `Executor` abstractions.
- Added a basic PyTorch training step registered as `"torch::train"`, along with other registrable
  components, such as `Model`, `DataLoader`, `Sampler`, `DataCollator`, `Optimizer`, and `LRScheduler`.
- Added `DatasetRemixStep` in `tango.steps`.
- Added module `tango.common.sequences`.
- Added `DatasetDict` class in `tango.common.dataset_dict`.
- Added [🤗 Datasets](https://github.com/huggingface/datasets) integration.
- Added command-line options to set log level or disable logging completely.

### Changed

- `Step.work_dir`, `Step.unique_id`, `Step.dependencies`, and `Step.recursive_dependencies`
  are now a properties instead of methods.
- `tango run` command will acquire a lock on the directory to avoid race conditions.
- Integrations can now be installed with `pip install tango[INTEGRATION_NAME]`. For example,
  `pip install tango[torch]`.
- Added method `Registrable.search_modules()` for automatically finding and importing the modules
  where a given ``name`` might be registered.
- `FromParams.from_params()` and `Registrable.resolve_class_name` will now call `Registrable.search_modules()` to automatically import modules where the type might be defined.
  Thus for classes that are defined and registered within any `tango.*` submodules it is not necessary to explicitly import them.

### Fixed

- `Step` implementations can now take arbitrary `**kwargs` in their `run()` methods.

## [v0.0.3](https://github.com/allenai/tango/releases/tag/v0.0.3) - 2021-09-27

### Added

- Added `tango` command.

## [v0.0.2](https://github.com/allenai/tango/releases/tag/v0.0.2) - 2021-09-27

### Added

- Ported over core tango components from AllenNLP.

## [v0.0.1](https://github.com/allenai/tango/releases/tag/v0.0.1) - 2021-09-22

### Added

- Added initial project boilerplate.
