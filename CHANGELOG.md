# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

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
