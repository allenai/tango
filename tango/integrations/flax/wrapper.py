from tango.common.registrable import Registrable
from abc import abstractmethod
from typing import Dict


class FlaxWrapper(Registrable):
    @abstractmethod
    def train_metrics(self, state, batch, labels) -> Dict:
        """
        Returns train metrics other than loss as Dict.
        """
        pass

    @abstractmethod
    def train_loss(self, params, state, batch, dropout_rng, labels):
        """
        This function performs the forward pass and computes loss. The function
        should return the loss for the batch as a jax device array. The gradient
        of this function is used for training.
        """
        pass

    @abstractmethod
    def val_metrics(self, batch, logits, labels) -> Dict:
        """
        Returns validation metrics as Dict.
        """
        pass

    @abstractmethod
    def eval_metrics(self, batch, logits, labels) -> Dict:
        """
        Returns evaluation metrics as  Dict.
        """
        pass
