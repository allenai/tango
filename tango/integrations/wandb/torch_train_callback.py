from typing import Any, Dict, List, Optional

import torch
import wandb

from tango.common.exceptions import ConfigurationError
from tango.integrations.torch.train_callback import TrainCallback
from tango.integrations.torch.util import peak_gpu_memory

from .util import check_environment
from .workspace import WandbWorkspace


@TrainCallback.register("wandb::log")
class WandbTrainCallback(TrainCallback):
    """
    A torch :class:`~tango.integrations.torch.TrainCallback` for use with
    the :class:`~tango.integrations.torch.TorchTrainStep` that logs training and
    validation metrics to W&B.

    This can be used with any :class:`~tango.workspace.Workspace` implementation,
    including :class:`WandbWorkspace`.

    .. tip::

        Registered as a :class:`~tango.integrations.torch.TrainCallback`
        under the name "wandb::log".

    .. important::

        When this callback is used with the :class:`WandbWorkspace` it will log metrics
        to the same W&B project that the workspace uses. The ``group`` and ``name``
        parameters will also automatically be set, so a :class:`~tango.common.exceptions.ConfigurationError`
        will be raised if any of ``project``, ``entity``, ``group``, or ``name`` are set in this callback.

    :param project:
        W&B project to associated this run with.

    :param entity:
        W&B entity (user or organization) to associated this run with.

    :param group:
        W&B group to associated this run with.

    :param name:
        Set the name of the run in W&B. If not set, the default will be the name of the step.

    :param notes:
        Arbitrary notes to add in W&B to this run.

    :param tags:
        Arbitrary tags to add in W&B to this run.

    :param watch_model:
        If ``True``, ``wandb.watch()`` is called to collect gradients and other information
        about the model throughout training.
        See `docs.wandb.ai/ref/python/watch <https://docs.wandb.ai/ref/python/watch>`_.

    :param wandb_config:
        Arbitrary configuration fields to set in W&B for this run.
        See `docs.wandb.ai/guides/track/config <https://docs.wandb.ai/guides/track/config>`_.

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
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if self.is_local_main_process:
            check_environment()

        if isinstance(self.workspace, WandbWorkspace) or wandb.run is not None:
            err_msg_template = "Cannot set '{var_name}' in WandbTrainCallback "
            if isinstance(self.workspace, WandbWorkspace):
                err_msg_template += "since it has already been set from the WandbWorkspace."
            else:
                err_msg_template += "since a W&B run has already been initialized."
            for var, var_name in [
                (project, "project"),
                (entity, "entity"),
                (group, "group"),
                (name, "name"),
            ]:
                if var is not None:
                    raise ConfigurationError(err_msg_template.format(var_name=var_name))

        self.project = (
            project if not isinstance(self.workspace, WandbWorkspace) else self.workspace.project
        )
        self.entity = (
            entity if not isinstance(self.workspace, WandbWorkspace) else self.workspace.entity
        )
        self.group = group or self.step_id
        self.notes = notes or self._get_default_notes()
        self.tags = tags
        self.watch_model = watch_model

        self.wandb_config = self.train_config.as_dict()
        del self.wandb_config["worker_id"]
        if wandb_config is not None:
            self.wandb_config.update(wandb_config)
        if wandb.run is None:
            self.wandb_config["job_type"] = "train_metrics"

        self.run_name: str = name or self.step_name or "train"
        if self.train_config.is_distributed:
            self.run_name += f" (rank {self.train_config.worker_id})"

        self.run_id: str = (
            wandb.run.id  # type: ignore[attr-defined]
            if wandb.run is not None
            else self.step_id + f"-rank{self.train_config.worker_id}"
        )
        self.resume: Optional[str] = None
        self.should_finalize_run: bool = (
            wandb.run is None
        )  # if we have to start out own W&B run, we need to finish it

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.resume = "allow"

    def pre_train_loop(self) -> None:
        if wandb.run is None:
            if self.run_id is None:
                self.run_id = self.step_id + f"-rank{self.train_config.worker_id}"
            # Initialize a new W&B run.
            wandb.init(
                id=self.run_id,
                dir=str(self.work_dir),
                project=self.project,
                entity=self.entity,
                group=self.group,
                name=self.run_name,
                notes=self.notes,
                config=self.wandb_config,
                tags=self.tags,
                job_type="train_metrics",
            )
        else:
            # We are already running inside of a W&B run, possibly because
            # we're using the WandbWorkspace.
            wandb.config.update(self.wandb_config)
            if self.tags:
                wandb.run.tags = (wandb.run.tags or tuple()) + tuple(self.tags)
            if self.notes:
                wandb.run.notes = self.notes

        if self.watch_model:
            wandb.watch(self.training_engine.model)

        # Log GPU memory statistics.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        peak_gpu_mbs = peak_gpu_memory()
        if self.is_local_main_process:
            metrics = {f"sys/worker{rank}_peak_gpu_mem": mbs for rank, mbs in peak_gpu_mbs.items()}
            metrics["epoch"] = 0
            wandb.log(metrics, step=0)

    def post_train_loop(self, step: int, epoch: int) -> None:
        if self.should_finalize_run:
            wandb.finish()

    def log_batch(self, step: int, epoch: int, batch_loss: float) -> None:
        peak_gpu_mbs = peak_gpu_memory()
        if self.is_local_main_process:
            metrics = {
                "train/loss": batch_loss,
                "train/lr": self.training_engine.optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            }
            metrics.update(
                {f"sys/worker{rank}_peak_gpu_mem": mbs for rank, mbs in peak_gpu_mbs.items()}
            )
            wandb.log(
                metrics,
                step=step + 1,
            )

    def post_val_loop(
        self, step: int, epoch: int, val_metric: float, best_val_metric: float
    ) -> None:
        if self.is_local_main_process:
            wandb.log(
                {
                    f"val/{self.train_config.val_metric_name}": val_metric,
                    f"val/best_{self.train_config.val_metric_name}": best_val_metric,
                    "epoch": epoch,
                },
                step=step + 1,
            )

    def _get_default_notes(self) -> str:
        notes = (
            f'Metrics for Tango step "{self.step_name}" from worker {self.train_config.worker_id}.'
        )
        if isinstance(self.workspace, WandbWorkspace):
            notes += f"\nMain run for step: {self.workspace.wandb_project_url}/runs/{self.step_id}/overview"
        return notes
