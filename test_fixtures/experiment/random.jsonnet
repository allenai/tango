{
    "steps": {
        "rand_string1": {"type": "random_string", "length": 5},
        "rand_string2": {"type": "random_string", "length": 5},
        "string1": {
            "type": "concat_strings",
            "string1": {"type": "ref", "ref": "rand_string1"},
            "string2": {"type": "ref", "ref": "rand_string2"},
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
