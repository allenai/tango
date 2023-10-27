{
    "steps": {
        "add_numbers": {
            "type": "addition",
            "num1": 34,
            "num2": 8
        },
        "multiply_result": {
            "type": "scale_up",
            "num1": {"type": "ref", "ref": "add_numbers"},
            "factor": 10,
        },
        "divide_result": {
            "type": "scale_down",
            "num1": {"type": "ref", "ref": "multiply_result"},
            "factor": 5,
        },
        "add_x": {
            "type": "addition",
            "num1": {"type": "ref", "ref": "divide_result"},
            "num2": 1,
        },
        "print": {
            "type": "print",
            "num": {"type": "ref", "ref": "add_x"},
        },
    },
}
