local i = [0.0, 1.0];
local pi = [3.1415926535, 0.0];

{
    "steps": {
        "cos": {
            "type": "ccos",
            "x": pi
        },
        "sin": {
            "type": "csin",
            "x": pi
        },
        "i_times_sin": {
            "type": "cmul",
            "a": i,
            "b": { "type": "ref", "ref": "sin" }
        },
        "sum": {
            "type": "cadd",
            "a": { "type": "ref", "ref": "cos" },
            "b": { "type": "ref", "ref": "i_times_sin" },
        },

        "i_times_pi": {
            "type": "cmul",
            "a": i,
            "b": pi
        },
        "pow_e": {
            "type": "cexp",
            "x": { "type": "ref", "ref": "i_times_pi" }
        },

        "sub": {
            "type": "csub",
            "a": { "type": "ref", "ref": "sum" },
            "b": { "type": "ref", "ref": "pow_e" },
        },

        "print": {
            "type": "print",
            "input": { "type": "ref", "ref": "sub" }
        }
    }
}