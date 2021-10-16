import os
from typing import Optional, Dict, Any, List
import sys

from overrides import overrides

from tango.integrations.torch.train_callback import TrainCallback


@TrainCallback.register("wandb::log")
class WandbTrainCallback(TrainCallback):
    """
    A torch :class:`~tango.integrations.torch.TrainCallback` for use with
    the :class:`~tango.integrations.torch.TorchTrainStep` that logs training and
    validation metrics to W&B.

    .. tip::

        Registered as a :class:`~tango.integrations.torch.TrainCallback`
        under the name "wandb::log".

    .. important::
        This class also requires the :mod:`~tango.integrations.torch` integration.

    Parameters
    ----------

    project : :class:`str`
        W&B project to associated this run with.
    entity : :class:`str`
        W&B entity (user or organization) to associated this run with.
    group : :class:`str`
        W&B group to associated this run with.
    name : :class:`str`
        Set the name of the run in W&B.
    notes : :class:`str`
        Arbitrary notes to add in W&B to this run.
    tags : ``List[str]``
        Arbitrary tags to add in W&B to this run.
    watch_model : :class:`bool`
        If ``True``, ``wandb.watch()`` is called to collect gradients and other information
        about the model throughout training.
        See `docs.wandb.ai/ref/python/watch <https://docs.wandb.ai/ref/python/watch>`_.
    wandb_config : ``Dict[str, Any]``
        Arbitrary configuration fields to set in W&B for this run.
        See `docs.wandb.ai/guides/track/config <https://docs.wandb.ai/guides/track/config>`_.
    wandb_kwargs : ``Dict[str, Any]``
        Additional keyword arguments that will be passed to ``wandb.init()``.
        See `docs.wandb.ai/ref/python/init <https://docs.wandb.ai/ref/python/init>`_.

    """

    def __init__(
        self,
        *args,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        group: Optional[str] = None,
        name: Optional[str] = None,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
        watch_model: bool = False,
        wandb_config: Optional[Dict[str, Any]] = None,
        wandb_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if self.is_local_main_process and "WANDB_API_KEY" not in os.environ:
            print(
                "Missing environment variable 'WANDB_API_KEY' required to authenticate to Weights & Biases.",
                file=sys.stderr,
            )

        self._watch_model = watch_model
        self._run_id: Optional[str] = None
        self._wandb_kwargs: Dict[str, Any] = dict(
            dir=str(self.work_dir),
            project=project,
            entity=entity,
            group=group,
            name=name,
            notes=notes,
            config=wandb_config,
            tags=tags,
            anonymous="allow",
            **(wandb_kwargs or {}),
        )

    @overrides
    def state_dict(self) -> Dict[str, Any]:
        if self.is_local_main_process:
            return {
                "run_id": self._run_id,
            }
        else:
            return {}

    @overrides
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self.is_local_main_process:
            self._wandb_kwargs["resume"] = "auto"
            self._run_id = state_dict["run_id"]

    @overrides
    def pre_train_loop(self) -> None:
        if self.is_local_main_process:
            import wandb

            self.wandb = wandb

            if self._run_id is None:
                self._run_id = self.wandb.util.generate_id()

            self.wandb.init(id=self._run_id, **self._wandb_kwargs)

            if self._watch_model:
                self.wandb.watch(self.model)

    @overrides
    def post_train_loop(self) -> None:
        if self.is_local_main_process:
            self.wandb.finish()

    @overrides
    def post_batch(self, step: int, batch_loss: float) -> None:
        if self.is_local_main_process:
            self.wandb.log(
                {"train/loss": batch_loss, "train/lr": self.optimizer.param_groups[0]["lr"]},
                step=step,
            )

    @overrides
    def post_val_loop(self, step: int, val_metric_name: str, val_metric: float) -> None:
        if self.is_local_main_process:
            self.wandb.log({f"val/{val_metric_name}": val_metric}, step=step)
