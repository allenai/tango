from abc import abstractmethod
from typing import Dict

from tango.common.registrable import Registrable


class FlaxWrapper(Registrable):
    """
    A wrapper class which contains functions that need to be defined by the user
    for using the ``flax::train`` and ``flax::eval`` steps.
    """

    def train_metrics(self, state, batch, labels) -> Dict:
        """
        Returns the train metrics other than loss as Dict.
        """
        # return empty dict if no other metrics to compute
        return {}

    @abstractmethod
    def train_loss(self, params, state, batch, dropout_rng, labels):
        """
        This function performs the forward pass and computes loss. The function
        should return the loss for the batch as a jax device array. The gradient
        of this function is used for training.
        """
        raise NotImplementedError()

    @abstractmethod
    def val_metrics(self, batch, logits, labels) -> Dict:
        """
        Returns the validation metrics as Dict.
        """
        raise NotImplementedError()

    @abstractmethod
    def eval_metrics(self, batch, logits, labels) -> Dict:
        """
        Returns the evaluation metrics as  Dict.
        """
        raise NotImplementedError()
