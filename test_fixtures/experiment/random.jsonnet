{
    "steps": {
        "string1": {
            "type": "concat_strings",
            "string1": {"type": "random_string"},
            "string2": {"type": "random_string"},
        },
        "string2": {
            "type": "string",
            "result": "foo",
        },
        "final_string": {
            "type": "concat_strings",
            "string1": {"type": "ref", "ref": "string1"},
            "string2": {"type": "ref", "ref": "string2"},
        }
    }
}
