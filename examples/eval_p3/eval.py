import logging
from dataclasses import dataclass
from typing import Union

from rouge_score.scoring import Score

from tango import Format, JsonFormat, Step
from tango.common import DatasetDict
from tango.common.logging import make_tqdm

logger = logging.getLogger(__name__)
tqdm = make_tqdm(logger)


@dataclass
class RougeScore:
    precision: float
    recall: float
    fmeasure: float

    def __add__(self, other: Union[Score, "RougeScore"]) -> "RougeScore":
        return RougeScore(
            self.precision + other.precision,
            self.recall + other.recall,
            self.fmeasure + other.fmeasure,
        )

    def __sub__(self, other: Union[Score, "RougeScore"]) -> "RougeScore":
        return RougeScore(
            self.precision - other.precision,
            self.recall - other.recall,
            self.fmeasure - other.fmeasure,
        )

    def __mul__(self, other: float) -> "RougeScore":
        return RougeScore(self.precision * other, self.recall * other, self.fmeasure * other)

    def __truediv__(self, other: float) -> "RougeScore":
        return RougeScore(self.precision / other, self.recall / other, self.fmeasure / other)

    def __iadd__(self, other: Union[Score, "RougeScore"]) -> "RougeScore":
        self.precision += other.precision
        self.recall += other.recall
        self.fmeasure += other.fmeasure
        return self

    def __isub__(self, other: Union[Score, "RougeScore"]) -> "RougeScore":
        self.precision -= other.precision
        self.recall -= other.recall
        self.fmeasure -= other.fmeasure
        return self

    def __imul__(self, other: float) -> "RougeScore":
        self.precision *= other
        self.recall *= other
        self.fmeasure *= other
        return self

    def __itruediv__(self, other: float) -> "RougeScore":
        self.precision /= other
        self.recall /= other
        self.fmeasure /= other
        return self


@dataclass
class RougeResults:
    rouge1: RougeScore
    rouge2: RougeScore
    rougeL: RougeScore


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

        rouge1 = RougeScore(0, 0, 0)
        rouge2 = RougeScore(0, 0, 0)
        rougeL = RougeScore(0, 0, 0)
        for instance in tqdm(input[input_split], desc="Calculating scores"):
            target = instance[target_field]
            instance_rouge1 = RougeScore(0, 0, 0)
            instance_rouge2 = RougeScore(0, 0, 0)
            instance_rougeL = RougeScore(0, 0, 0)
            for prediction in instance[prediction_field]:
                scores = scorer.score(target, prediction)
                instance_rouge1 += scores["rouge1"]
                instance_rouge2 += scores["rouge2"]
                instance_rougeL += scores["rougeL"]
            num_predictions = len(instance[prediction_field])
            rouge1 += instance_rouge1 / num_predictions
            rouge2 += instance_rouge2 / num_predictions
            rougeL += instance_rougeL / num_predictions

        length = len(input[input_split])
        return RougeResults(rouge1 / length, rouge2 / length, rougeL / length)
