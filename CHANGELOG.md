# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Added the ability to track a metric other than "loss" for validation in `TorchTrainStep` ("torch::train").

### Fixed

- Final model returned from `TorchTrainStep` ("torch::train") will have best weights loaded.
- Checkpoints are saved from `TorchTrainStep` ("torch::train") even when there is no validation loop.

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
