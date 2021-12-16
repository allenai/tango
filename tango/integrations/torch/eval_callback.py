from pathlib import Path
from typing import Any, Dict

from tango.common.dataset_dict import DatasetDictBase
from tango.common.registrable import Registrable

from .data import DataLoader
from .model import Model


class EvalCallback(Registrable):
    """
    An ``EvalCallback`` is a :class:`~tango.common.Registrable` class that can be used
    within :class:`TorchEvalStep` to customize the behavior of the evaluation loop,
    similar to how :class:`TrainCallback` is used to customize the behavior of the training
    loop.

    .. tip::
        All of the parameters to this base class will be automatically set within
        the training loop, so you shouldn't include them in your config for your callbacks.

    Attributes
    ----------
    step_id : :class:`str`
    work_dir : :class:`~pathlib.Path`
    model : :class:`Model`
    dataset_dict : :class:`tango.common.DatasetDictBase`
    dataloader : :class:`DataLoader`

    """

    def __init__(
        self,
        step_id: str,
        work_dir: Path,
        model: Model,
        dataset_dict: DatasetDictBase,
        dataloader: DataLoader,
    ) -> None:
        self.step_id = step_id
        self.work_dir = work_dir
        self.model = model
        self.dataset_dict = dataset_dict
        self.dataloader = dataloader

    def pre_eval_loop(self) -> None:
        """
        Called right before the first batch is processed.
        """
        pass

    def post_eval_loop(self, aggregated_metrics: Dict[str, float]) -> None:
        """
        Called after the evaluation loop completes with the final aggregated metrics.

        This is the last method that is called, so any cleanup can be done in this method.
        """
        pass

    def pre_batch(self, step: int, batch: Dict[str, Any]) -> None:
        """
        Called directly before processing a batch.
        """
        pass

    def post_batch(self, step: int, batch_outputs: Dict[str, Any]) -> None:
        """
        Called directly after processing a batch with the outputs of the batch.

        .. tip::
            This method can be used to modify ``batch_outputs`` in place, which is useful
            in scenarios where you might need to aggregate metrics
            in a special way other than a simple average. If that's the case, make sure
            to set ``auto_aggregate_metrics`` to ``False`` in :class:`TorchEvalStep`.

        """
        pass
