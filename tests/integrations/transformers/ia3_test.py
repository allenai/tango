import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tango.integrations.transformers.ia3 import GPT_2_IA3_CONFIG, modify_with_ia3


def test_ia3():

    config = GPT_2_IA3_CONFIG
    model_name = "sshleifer/tiny-gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_seq = tokenizer(["A tiny test on a tiny model."], return_tensors="pt")

    model = AutoModelForCausalLM.from_pretrained(model_name)

    with torch.no_grad():
        old_outputs = model(
            input_ids=input_seq.input_ids,
            attention_mask=input_seq.attention_mask,
            labels=input_seq.input_ids,
        )

    model = modify_with_ia3(model, config=config)

    with torch.no_grad():
        new_outputs = model(
            input_ids=input_seq.input_ids,
            attention_mask=input_seq.attention_mask,
            labels=input_seq.input_ids,
        )

    logits_diff = torch.abs(old_outputs.logits - new_outputs.logits).mean()
    assert logits_diff < 1e-10

    loss_diff = torch.abs(old_outputs.loss - new_outputs.loss)
    assert loss_diff < 1e-10
