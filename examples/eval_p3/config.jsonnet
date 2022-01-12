local model = "gpt2";
local batch_size = 8;

local datasets = [
    'xsum_DOC_boils_down_to_simple_idea_that',
    'xsum_DOC_given_above_write_one_sentence',
    'xsum_DOC_how_would_you_rephrase_few_words',
    'xsum_DOC_tldr',
    'xsum_DOC_write_summary_of_above',
    'xsum_article_DOC_summary',
    'xsum_college_roommate_asked_DOC_so_I_recap',
    'xsum_read_below_DOC_write_abstract',
    'xsum_summarize_DOC',
    'xsum_summarize_this_DOC_summary'
];

local dataset_steps = std.foldl(
    function(x, dataset_name) x + {
        ["dataset_" + dataset_name]: {
            "type": "datasets::load",
            "path": "bigscience/P3",
            "name": dataset_name,
        },
        ["generation_" + dataset_name]: {
            "type": "transformers::run_generation_dataset",
            "max_length": 200,
            "input": {"ref": "dataset_" + dataset_name},
            "batch_size": batch_size,
            "model_name": model,
            "prompt_field": "inputs_pretokenized",
            "output_field": "generation",
            "splits": ["validation"]
        },
        ["eval_" + dataset_name]: {
            "type": "rouge_score",
            "input": {"ref": "generation_" + dataset_name},
            "input_split": "validation",
            "target_field": "targets_pretokenized",
            "prediction_field": "generation"
        }
    },
    datasets,
    {}
);

{
    "steps": dataset_steps + {
        "all_generations": {
            "type": "dataset_combine",
            "inputs": std.map(
                function(dataset_name) {"ref": "generation_" + dataset_name},
                datasets
            )
        },
        "all_evaluations": {
            "type": "rouge_score",
            "input": {"ref": "all_generations"},
            "input_split": "validation",
            "target_field": "targets_pretokenized",
            "prediction_field": "generation"
        }
    }
}
