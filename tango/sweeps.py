import itertools
from collections import OrderedDict
from dataclasses import dataclass

from tango.common import Params, Registrable
from tango.common.testing import run_experiment

main_config_path = (
    "/Users/sabhyac/Desktop/sabhya/tango/test_fixtures/sweeps/basic_test/config.jsonnet"
)
sweeps_config_path = (
    "/Users/sabhyac/Desktop/sabhya/tango/test_fixtures/sweeps/basic_test/sweeps-config.jsonnet"
)
components = (
    "/Users/sabhyac/Desktop/sabhya/tango/test_fixtures/sweeps/basic_test/basic_arithmetic.py"
)


class Sweeper(Registrable):
    def __init__(self, main_config_path: str, sweeps_config_path: str, components: str):
        super(Registrable, self).__init__()
        self.main_config_path = main_config_path
        self.sweep_config = load_config(sweeps_config_path)
        self.main_config_path = main_config_path
        self.components = components

    # returns all the combinations of hyperparameters in the form of a list of lists
    def get_combinations(self):
        hyperparams = self.sweep_config.config["hyperparameters"]
        hyperparams_lsts = []
        for val in hyperparams.values():
            hyperparams_lsts.append(val)
        hyperparam_combos = list(itertools.product(*hyperparams_lsts))
        return hyperparam_combos

    # TODO: trying to figure the best path forward? should i use tests?
    def run_experiments(self):
        hyperparam_combos = self.get_combinations()
        for combination in hyperparam_combos:
            main_config = self.override_hyperparameters(combination)
            # TODO: need to figure where & how to store results / way to track runs
            with run_experiment(main_config, include_package=[self.components]) as run_dir:
                # TODO: fill in something here?
                pass

    # TODO: wondering if this function should be here or in a test_file?
    def override_hyperparameters(self, experiment_tuple: tuple):
        # Override all the hyperparameters in the current experiment_config
        overrides = {}
        for (i, key) in enumerate(self.sweep_config.config["hyperparameters"].keys()):
            overrides[key] = experiment_tuple[i]
        # load the config & override it
        main_config = Params.from_file(self.main_config_path, params_overrides=overrides)
        return main_config


# function that loads the config from a specified yaml or jasonnet file
# TODO: how do I read "wandb" form config and call appropriate class
def load_config(sweeps_config_path: str):
    return SweepConfig.from_file(sweeps_config_path)


# data class that loads the parameters
# TODO: unsure about how to specify a default here?
@dataclass(frozen=True)
class SweepConfig(Params):
    config: OrderedDict
