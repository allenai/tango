from abc import abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from tango.common.dataset_dict import DatasetDictBase
from tango.common.exceptions import ConfigurationError
from tango.common.lazy import Lazy
from tango.common.registrable import Registrable
from tango.common.tqdm import Tqdm
from tango.format import Format, JsonFormat
from tango.step import Step

from .data import FlaxDataLoader
from .eval_callback import EvalCallback
from .model import Model
from .util import get_PRNGkey


class FlaxEvalWrapper(Registrable):
    @abstractmethod
    def eval_step(self, state, batch):
        """
        returns logits and metrics.
        """
        pass


@Step.register("flax::eval")
class FlaxEvalStep(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT: Format = JsonFormat()
    SKIP_ID_ARGUMENTS = {"log_every"}

    def run(  # type: ignore[override]
        self,
        model: Model,
        state: TrainState,
        dataset: DatasetDictBase,
        dataloader: Lazy[FlaxDataLoader],
        eval_wrapper: FlaxEvalWrapper,
        test_split: str = "test",
        seed: int = 42,
        log_every: int = 1,
        do_distributed: bool = False,
        eval_steps: Optional[int] = None,
        metric_names: Sequence[str] = ("loss",),
        auto_aggregate_metrics: bool = True,
        callbacks: Optional[List[Lazy[EvalCallback]]] = None,
    ) -> Dict[str, float]:

        # construct dataloader
        dataloader: FlaxDataLoader = dataloader.construct(dataset=dataset[test_split])

        steps: int
        try:
            dataloader_len = dataloader.dataset_size
            steps = dataloader_len if eval_steps is None else min(dataloader_len, eval_steps)
        except TypeError:
            if eval_steps is None:
                raise ConfigurationError(
                    "You must set 'eval_steps' for streaming/iterable datasets"
                )
            else:
                steps = eval_steps

        if do_distributed:
            devices = jax.devices()
            if len(devices) <= 1:
                raise ConfigurationError(
                    "YOu have set distributed training=True but there is only one device."
                )

        # Initialize callbacks
        callbacks_: List[EvalCallback] = [
            callback.construct(
                step_id=self.unique_id,
                work_dir=self.work_dir,
                model=model,
                dataset_dict=dataset,
                dataloader=dataloader,
            )
            for callback in (callbacks or [])
        ]

        for callback in callbacks_:
            callback.pre_eval_loop()

        eval_metrics: Dict = defaultdict(list)
        aggregated_metrics: Dict = defaultdict(float)

        rng = get_PRNGkey()
        start_step = 0
        batches = Tqdm.tqdm(
            dataloader(rng, do_distributed), initial=start_step, total=steps, desc="Evaluating"
        )
        for step, batch in enumerate(batches):
            for callback in callbacks_:
                callback.pre_batch(step, batch)

            if do_distributed:
                parallel_eval_step = jax.pmap(eval_wrapper.eval_step, axis_name="batch")
                logits, metrics = parallel_eval_step(state, batch)
                metrics = jax.lax.pmean(metrics, axis_name="batch")
            else:
                logits, metrics = eval_wrapper.eval_step(state, batch)
            metrics = jax.device_get(metrics)
            metrics = jax.tree_map(lambda x: x.item(), metrics)

            for callback in callbacks_:
                callback.post_batch(step, logits)

            for key, value in metrics.items():
                eval_metrics[key].append(value)

        for key, val in eval_metrics.items():
            aggregated_metrics[key] = jax.tree_map(jnp.mean, jnp.array(val)).item()

        print("Test metrics: " , aggregated_metrics)

        for callback in callbacks_:
            callback.post_eval_loop(aggregated_metrics)

        return aggregated_metrics
