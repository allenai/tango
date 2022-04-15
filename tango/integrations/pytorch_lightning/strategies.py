import pytorch_lightning as pl
import pytorch_lightning.strategies

from tango.common.registrable import Registrable


class LightningStrategy(pytorch_lightning.strategies.Strategy, Registrable):
    """
    This is simply a :class:`~tango.common.registrable.Registrable`
    version of the PyTorch Lightning :class:`~pytorch_lightning.strategies.Strategy`.
    """


# Register all strategies.
for name, strategy in pl.strategies.StrategyRegistry.items():

    def make_strategy(*args, **kwargs):
        kwargs = {**strategy["init_params"], **kwargs}
        return strategy["strategy"](*args, **kwargs)

    LightningStrategy.register("pytorch_lightning::" + name)(make_strategy)  # type: ignore
