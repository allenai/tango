import transformers

from tango.integrations.transformers import add_soft_prompt


def test_soft_prompt():
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small")
    prompt = "translate English to German: That is good."
    model.eval()
    generated = model.generate(
        tokenizer.encode(prompt, return_tensors="pt"), num_beams=10, num_return_sequences=5
    )
    original_output = [tokenizer.decode(g) for g in generated]

    add_soft_prompt(model, prompt_length=3)
    model.eval()
    generated = model.generate(
        tokenizer.encode(prompt, return_tensors="pt"), num_beams=10, num_return_sequences=5
    )
    prompted_output = [tokenizer.decode(g) for g in generated]

    assert original_output != prompted_output


def test_soft_prompt_twice():
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    add_soft_prompt(model, prompt_length=2)
    model.eval()
    generated = model.generate(tokenizer.encode("It was the best of times.", return_tensors="pt"))
    prompted_output1 = tokenizer.decode(generated[0])

    add_soft_prompt(model, prompt_length=5)
    model.eval()
    generated = model.generate(tokenizer.encode("It was the best of times.", return_tensors="pt"))
    prompted_output2 = tokenizer.decode(generated[0])

    assert prompted_output1 != prompted_output2
