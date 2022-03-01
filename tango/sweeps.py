from dataclasses import dataclass, asdict
from tango.common import Params
from tango.common import Registrable
import itertools

config_path = "/Users/sabhyac/Desktop/sabhya/tango/tango/sweep-config.yaml"

# main sweeper class
class Sweeper(Registrable):
    def __init__(self, config_path):
        super(Registrable, self).__init__()
        self.config_path = config_path
        self.sweep_config = load_config(config_path)
    
    # returns all the combinations of hyperparameters in the form of a list of lists
    def get_combinations(self):
        hyperparams = self.sweep_config.config["hyperparameters"]
        hyperparams_lsts = []
        for key, val in hyperparams.items():
            hyperparams_lsts.append(val)            
        hyperparam_combos = list(itertools.product(*hyperparams_lsts))
        return hyperparam_combos
    
    # TODO: wondering if this function should be here or in a test_file?
    def override_hyperparameters(self):
        pass
    
    # TODO: trying to figure the best path forward? should i use tests?
    def run_experiments(self):
        pass

# wandb sweeper class
@Sweeper.register("wandb")
class WandbSweeper(Sweeper):
    pass

# function that loads the config from a specified yaml or jasonnet file
# TODO: how do I read "wandb" form config and call appropriate class
def load_config(config_path: str):
    return SweepConfig.from_file(config_path)

# data class that loads the parameters
# TODO: unsure about how to specify a default here?
@dataclass(frozen=True)
class SweepConfig(Params):
    config: dict


if __name__ == "__main__":
    sw = Sweeper(config_path=config_path)
    print(sw.sweep_config)