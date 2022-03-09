{
    "steps": {
        "stringA": {"type": "logging-step", "string": "This is a logging test.", "num_log_lines": 5},
        "stringB": {
            "type": "concat_strings",
            "string1": {"type": "ref", "ref": "stringA"},
            "string2": "This is being logged."
        },
        "stringC": {"type": "logging-step", "string": "This is also a logging test.", "num_log_lines": 5},
        "final_string": {
            "type": "logging-step",
            "string": {"type": "ref", "ref": "stringB"},
            "num_log_lines": 3
        },
        "multiprocessing_result": {
            "type": "multiprocessing_step",
        }
    }
}