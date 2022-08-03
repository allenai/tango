import logging
from collections import defaultdict
from itertools import islice
from typing import Dict, List, Optional, Sequence

import jax
from flax import jax_utils
from flax.training.train_state import TrainState

from tango.common.dataset_dict import DatasetDictBase
from tango.common.exceptions import ConfigurationError
from tango.common.lazy import Lazy
from tango.common.tqdm import Tqdm
from tango.format import Format, JsonFormat
from tango.step import Step

from .data import FlaxDataLoader
from .eval_callback import EvalCallback
from .util import get_PRNGkey
from .wrapper import FlaxWrapper


@Step.register("flax::eval")
class FlaxEvalStep(Step):
    """
    A Flax evaluation loop that pairs well with :class:`FlaxTrainStep`.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name "flax::eval".

    .. important::

        The evaluation loop will use a GPU/TPU automatically if one is available.
        You can control which GPU it uses with the environment variable ``CUDA_VISIBLE_DEVICES``.
        For example, set ``CUDA_VISIBLE_DEVICES=1`` to force ``FlaxEvalStep`` to only use
        the GPU with ID 1.

    .. warning::

        By default the metrics specified by the ``metric_names`` parameter
        are aggregated by simply averaging across batches.
        This behavior is usually correct for metrics like "loss" or "accuracy",
        for example, but may not be correct for other metrics like "F1".

        If this is not correct for your metric you will need to handle the aggregation
        internally in your model or with an :class:`EvalCallback`
        using the :meth:`EvalCallback.post_batch()` method.
        Then set the parameter ``auto_aggregate_metrics`` to ``False``.

    """

    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT: Format = JsonFormat()
    SKIP_ID_ARGUMENTS = {"log_every"}

    def run(  # type: ignore[override]
        self,
        state: TrainState,
        dataset: DatasetDictBase,
        dataloader: Lazy[FlaxDataLoader],
        wrapper: FlaxWrapper,
        test_split: str = "test",
        seed: int = 42,
        log_every: int = 1,
        do_distributed: bool = False,
        eval_steps: Optional[int] = None,
        metric_names: Sequence[str] = ("loss",),
        auto_aggregate_metrics: bool = True,
        callbacks: Optional[List[Lazy[EvalCallback]]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the ``model``.

        :param state:
            The state of the model to evaluate. This contains the parameters.
        :param dataset:
            Should contain the test data.
        :param dataloader:
            The data loader that generates test batches. The batches should be :class:`dict`
            objects.
        :param wrapper:
            The wrapper should define :meth:`eval_metrics`.
        :param test_split:
            The name of the data split used for evaluation in the ``dataset_dict``.
            Default is "test".
        :param seed:
             Used to set the PRNG states at the beginning of the evaluation loop.
        :param log_every:
            Log every this many steps. Default is ``1``.
        :param do_distributed:
            Whether to do distributed training or not. Set as 0 or 1.
        :param eval_steps:
            The number of steps to evaluate for. If not specified evaluation will
            stop after a complete iteration through the ``dataloader``.
        :param metric_names:
            The names of the metrics to track and aggregate. Default is ``("loss",)``.
        :param auto_aggregate_metrics:
            If ``True`` (the default), the metrics will be averaged across batches.
            This may not be the correct behavior for some metrics (such as F1),
            in which you should set this to ``False`` and handle the aggregation
            internally in your model or with an :class:`EvalCallback`
            (using :meth:`EvalCallback.post_batch()`).
        :param callbacks:
            A list of :class:`EvalCallback`.

        """

        logger = logging.getLogger(FlaxEvalStep.__name__)
        # construct dataloader
        dataloader: FlaxDataLoader = dataloader.construct(
            dataset=dataset[test_split].set_format("numpy")
        )

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
        callbacks: List[EvalCallback] = [
            callback.construct(
                step_id=self.unique_id,
                work_dir=self.work_dir,
                dataset_dict=dataset,
                dataloader=dataloader,
            )
            for callback in (callbacks or [])
        ]

        for callback in callbacks:
            callback.pre_eval_loop()

        rng = get_PRNGkey(seed)
        devices = jax.devices()
        if len(devices) > 1:
            do_distributed = True

        def eval_step(state, batch):
            labels = batch.pop("labels")
            logits = state.apply_fn(**batch, params=state.params, train=False)[0]
            metrics = wrapper.eval_metrics(batch, logits, labels)
            if do_distributed:
                metrics = jax.lax.pmean(metrics, axis_name="batch")
            return logits, metrics

        if do_distributed:
            state = jax_utils.replicate(state)
            parallel_eval_step = jax.pmap(eval_step, axis_name="batch")

        eval_batches = enumerate(islice(dataloader(rng, do_distributed), steps))

        running_metrics: Dict[str, float] = defaultdict(float)
        aggregated_metrics: Dict[str, float] = defaultdict(float)

        with Tqdm.tqdm(eval_batches, desc="Evaluating", total=steps) as batch_iter:
            for step, batch in batch_iter:
                should_log_this_step = step % log_every == 0 or step == steps - 1
                for callback in callbacks:
                    callback.pre_batch(step, batch)

                if do_distributed:
                    logits, metrics = parallel_eval_step(state, batch)
                    metrics = jax_utils.unreplicate(metrics)
                else:
                    logits, metrics = eval_step(state, batch)

                for callback in callbacks:
                    callback.post_batch(step, logits)

                if auto_aggregate_metrics:
                    for key, val in metrics.items():
                        if key in metric_names:
                            running_metrics[key] += metrics[key].item()
                            aggregated_metrics[key] = running_metrics[key] / (step + 1)
                else:
                    aggregated_metrics.update(metrics)

                if should_log_this_step:
                    batch_iter.set_postfix(**aggregated_metrics)
                del batch

        logger.info("Evaluation Metrics:")
        for key, val in aggregated_metrics.items():
            logger.info(key, ":", val)

        for callback in callbacks:
            callback.post_eval_loop(aggregated_metrics)

        return aggregated_metrics
