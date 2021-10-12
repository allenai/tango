# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

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
