local i = [0.0, 1.0];
local pi = [3.1415926535, 0.0];

{
    "steps": {
        "i_times_pi": {
            "type": "cmul",
            "a": i,
            "b": pi
        },
        "pow_e": {
            "type": "cexp",
            "x": { "type": "ref", "ref": "i_times_pi" }
        },
        "plus_one": {
            "type": "cadd",
            "a": { "type": "ref", "ref": "pow_e" },
            "b": [1, 0]
        },
        "print": {
            "type": "print",
            "input": { "type": "ref", "ref": "plus_one" }
        }
    }
}