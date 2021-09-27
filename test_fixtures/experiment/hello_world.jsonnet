{
    "steps": {
        "hello": {"type": "string", "result": "Hello"},
        "hello_world": {
            "type": "concat_strings",
            "string1": {"type": "ref", "ref": "hello"},
            "string2": "World!",
            "join_with": ", ",
        },
    },
}
