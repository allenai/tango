import transformers

from tango.integrations.transformers import add_soft_prompt


def test_soft_prompt():
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    generated = model.generate(tokenizer.encode("It was the best of times.", return_tensors="pt"))
    original_output = tokenizer.decode(generated[0])

    add_soft_prompt(model, prompt_length=3)
    generated = model.generate(tokenizer.encode("It was the best of times.", return_tensors="pt"))
    prompted_output = tokenizer.decode(generated[0])

    assert original_output != prompted_output


def test_soft_prompt_twice():
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    add_soft_prompt(model, prompt_length=2)
    generated = model.generate(tokenizer.encode("It was the best of times.", return_tensors="pt"))
    prompted_output1 = tokenizer.decode(generated[0])

    add_soft_prompt(model, prompt_length=5)
    generated = model.generate(tokenizer.encode("It was the best of times.", return_tensors="pt"))
    prompted_output2 = tokenizer.decode(generated[0])

    assert prompted_output1 != prompted_output2
