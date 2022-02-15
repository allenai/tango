from collections import defaultdict
from itertools import islice
from typing import Dict, List, Optional, Sequence

import torch

from tango.common.dataset_dict import DatasetDictBase
from tango.common.exceptions import ConfigurationError
from tango.common.lazy import Lazy
from tango.common.tqdm import Tqdm
from tango.format import Format, JsonFormat
from tango.step import Step

from .data import DataLoader
from .eval_callback import EvalCallback
from .model import Model
from .util import check_dataset, move_to_device, resolve_device, set_seed_all


@Step.register("torch::eval")
class TorchEvalStep(Step):
    """
    A PyTorch evaluation loop that pairs well with :class:`TorchTrainStep`.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name "torch::eval".

    .. important::

        The evaluation loop will use a GPU automatically if one is available.
        You can control which GPU it uses with the environment variable ``CUDA_VISIBLE_DEVICES``.
        For example, set ``CUDA_VISIBLE_DEVICES=1`` to force ``TorchEvalStep`` to only use
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
        model: Model,
        dataset_dict: DatasetDictBase,
        dataloader: Lazy[DataLoader],
        test_split: str = "test",
        seed: int = 42,
        eval_steps: Optional[int] = None,
        log_every: int = 1,
        metric_names: Sequence[str] = ("loss",),
        auto_aggregate_metrics: bool = True,
        callbacks: Optional[List[Lazy[EvalCallback]]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the ``model``.

        :param model:
            The model to evaluate. It should return a ``dict`` from its ``forward()`` method
            that includes all of the metrics in ``metric_names`` .
        :param dataset_dict:
            Should contain the test data.
        :param dataloader:
            The data loader that generates test batches. The batches should be :class:`dict`
            objects.
        :param test_split:
            The name of the data split used for evaluation in the ``dataset_dict``.
            Default is "test".
        :param seed:
            Used to set the RNG states at the beginning of the evaluation loop.
        :param eval_steps:
            The number of steps to evaluate for. If not specified evaluation will
            stop after a complete iteration through the ``dataloader``.
        :param log_every:
            Log every this many steps. Default is ``1``.
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
        set_seed_all(seed)

        check_dataset(dataset_dict, test_split)

        # Resolve device.
        device = resolve_device()

        # Prep model.
        model = model.eval().to(device)

        # Construct dataloader.
        dataloader: DataLoader = dataloader.construct(dataset=dataset_dict[test_split])

        steps: int
        try:
            dataloader_len = len(dataloader)
            steps = dataloader_len if eval_steps is None else min(dataloader_len, eval_steps)
        except TypeError:
            if eval_steps is None:
                raise ConfigurationError(
                    "You must set 'eval_steps' for streaming/iterable datasets"
                )
            else:
                steps = eval_steps

        # Initialize callbacks.
        callbacks: List[EvalCallback] = [
            callback.construct(
                step_id=self.unique_id,
                work_dir=self.work_dir,
                model=model,
                dataset_dict=dataset_dict,
                dataloader=dataloader,
            )
            for callback in (callbacks or [])
        ]
        for callback in callbacks:
            callback.pre_eval_loop()

        eval_batches = enumerate(islice(dataloader, steps))

        running_metrics: Dict[str, float] = defaultdict(float)
        aggregated_metrics: Dict[str, float] = {}

        with Tqdm.tqdm(eval_batches, desc="Evaluating", total=steps) as batch_iter:
            for step, batch in batch_iter:
                should_log_this_step = step % log_every == 0 or step == steps - 1

                for callback in callbacks:
                    callback.pre_batch(step, batch)

                batch = move_to_device(batch, device)
                with torch.inference_mode():
                    outputs = model(**batch)

                for callback in callbacks:
                    callback.post_batch(step, outputs)

                # Gather metrics we want to track.
                batch_metrics = {
                    k: outputs[k].item() if isinstance(outputs[k], torch.Tensor) else outputs[k]
                    for k in metric_names
                }

                # Aggregate metrics.
                if auto_aggregate_metrics:
                    for k in batch_metrics:
                        running_metrics[k] += batch_metrics[k]
                        aggregated_metrics[k] = running_metrics[k] / (step + 1)
                else:
                    aggregated_metrics.update(batch_metrics)

                # Update progress bar.
                if should_log_this_step:
                    batch_iter.set_postfix(**aggregated_metrics)

                # Clean up to help garbage collector. Hopefully this saves memory.
                del batch
                del outputs
                del batch_metrics

        for callback in callbacks:
            callback.post_eval_loop(aggregated_metrics)

        return aggregated_metrics
