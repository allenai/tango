import itertools
import json
import subprocess
from collections import OrderedDict
from dataclasses import dataclass

from tango.common import Params, Registrable


class Sweeper(Registrable):
    def __init__(
        self,
        main_config_path: str,
        sweeps_config_path: str,
        components: str,
    ):
        super(Registrable, self).__init__()
        self.main_config_path = main_config_path
        self.sweep_config = load_config(sweeps_config_path)
        self.main_config_path = main_config_path
        self.components = components

    # returns all the combinations of hyperparameters in the form of a list of lists
    def get_combinations(self) -> list:
        hyperparams = self.sweep_config.config["config"]["hyperparameters"]
        hyperparams_lsts = []
        for val in hyperparams.values():
            hyperparams_lsts.append(val)
        hyperparam_combos = list(itertools.product(*hyperparams_lsts))
        return hyperparam_combos

    # loops through all combinations of hyperparameters and creates a run for each
    def run_experiments(self):
        hyperparam_combos = self.get_combinations()
        for combination in hyperparam_combos:
            # main_config = self.override_hyperparameters(combination)
            overrides = self.override_hyperparameters(combination)
            # TODO: need to figure where & how to store results / way to track runs
            # specify what workspace to use
            subprocess.call(
                [
                    "tango",
                    "run",
                    self.main_config_path,
                    "--include-package",
                    self.components,
                    "--overrides",
                    json.dumps(overrides),
                ]
            )

    # function to override all the hyperparameters in the current experiment_config
    def override_hyperparameters(self, experiment_tuple: tuple) -> dict:
        overrides = {}
        for (i, key) in enumerate(self.sweep_config.config["config"]["hyperparameters"].keys()):
            overrides[key] = experiment_tuple[i]
        return overrides


# function that loads the config from a specified yaml or jasonnet file
def load_config(sweeps_config_path: str):
    return SweepConfig.from_file(sweeps_config_path)


# data class that loads the parameters
@dataclass(frozen=True)
class SweepConfig(Params):
    config: OrderedDict
