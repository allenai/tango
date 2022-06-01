import logging
from typing import Dict

from torch import Tensor
from torchmetrics.text.rouge import ROUGEScore

from tango import Format, JsonFormat, Step
from tango.common import DatasetDict
from tango.common.tqdm import Tqdm

logger = logging.getLogger(__name__)


@Step.register("rouge_score")
class RougeScoreStep(Step[Dict[str, Tensor]]):
    VERSION = "002"
    FORMAT: Format = JsonFormat()

    def run(  # type: ignore
        self,
        input: DatasetDict,
        input_split: str,
        target_field: str,
        prediction_field: str,
        use_stemmer: bool = True,
    ) -> Dict[str, Tensor]:
        metric = ROUGEScore(
            use_stemmer=use_stemmer,
            rouge_keys=("rouge1", "rouge2", "rougeL"),
            accumulate="avg",
        )

        for instance in Tqdm.tqdm(input[input_split], desc="Calculating scores"):
            target = instance[target_field]
            for prediction in instance[prediction_field]:
                metric.update(prediction, target)

        return metric.compute()
