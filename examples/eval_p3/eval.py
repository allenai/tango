import logging
from dataclasses import dataclass

from tango import Format, JsonFormat, Step
from tango.common import DatasetDict
from tango.common.logging import make_tqdm

logger = logging.getLogger(__name__)
tqdm = make_tqdm(logger)


@dataclass
class RougeResults:
    rouge1: float
    rouge2: float
    rougeL: float


@Step.register("rouge_score")
class RougeScoreStep(Step[RougeResults]):
    VERSION = "001"
    FORMAT: Format = JsonFormat()

    def run(  # type: ignore
        self,
        input: DatasetDict,
        input_split: str,
        target_field: str,
        prediction_field: str,
        use_stemmer: bool = True,
    ) -> RougeResults:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer)

        rouge1 = 0.0
        rouge2 = 0.0
        rougeL = 0.0
        for instance in tqdm(input[input_split], desc="Calculating scores"):
            scores = scorer.score(instance[target_field], instance[prediction_field])
            rouge1 += scores["rouge1"]
            rouge2 += scores["rouge2"]
            rougeL += scores["rougeL"]

        length = len(input[input_split])
        return RougeResults(rouge1 / length, rouge2 / length, rougeL / length)
