{
    "steps": {
        "strA": {
            "type": "sleep-print-maybe-fail",
            "string": "A",
            "seconds": 15,
        },
        "strB": {
            "type": "sleep-print-maybe-fail",
            "string": "B",
            "seconds": 15,
            "fail": false,
            "cache_results": false,
        },
        "concatenated": {
            "type": "concat_strings",
            "string1": {"type": "ref", "ref": "strA"},
            "string2": {"type": "ref", "ref": "strB"},
            "cache_results": false,
        }
    }
}