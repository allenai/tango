# Evaluating T0

This example uses the `transformers::run_generation_dataset` step to run the
[T0 model](https://api.semanticscholar.org/CorpusID:239009562). It runs the
[XSum summarization data](https://github.com/EdinburghNLP/XSum), prompted in 10 different ways, and computes
ROUGE scores for all variants. Finally, it computes an overall ROUGE score.

This example uses mostly built-in Tango steps. You will need the `datasets` and `transformers` integrations.
The only custom step in this example is the `RougeScoreStep`, which computes ROUGE scores from the
generated text.