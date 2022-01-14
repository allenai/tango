# Evaluating T0

This example uses the `run_generation_dataset` step to run the
[T0 model](https://api.semanticscholar.org/CorpusID:239009562). It runs the
[XSum summarization data](https://github.com/EdinburghNLP/XSum), prompted in 10 different ways, and computes
ROUGE scores for all variants. Finally, it computes an overall ROUGE score.