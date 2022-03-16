import logging
import random
from typing import Any, Dict, Optional

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from tango.integrations.torch import Model

logger = logging.getLogger(__name__)


def add_soft_prompt(model: Model, prompt_length: int, random_seed: int = 1940) -> None:
    """
    Takes a regular huggingface transformer, and equips it with a soft prompt.

    Example:

    .. testcode::
        import transformers

        model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        generated = model.generate(tokenizer.encode("It was the best of times.", return_tensors="pt"))
        original_output = tokenizer.decode(generated[0])

        add_soft_prompt(model, prompt_length=3)
        generated = model.generate(tokenizer.encode("It was the best of times.", return_tensors="pt"))
        prompted_output = tokenizer.decode(generated[0])

    :param model: the original huggingface transformer. This model is augmented in-place!
    :param prompt_length: the length of the soft prompt, in tokens

    """
    assert isinstance(model, PreTrainedModel)

    original_embedding: nn.Embedding = model.get_input_embeddings()  # type: ignore
    prompt_embedding = nn.Parameter(
        torch.empty(
            1,
            prompt_length,
            original_embedding.embedding_dim,
            dtype=original_embedding.weight.dtype,
            device=original_embedding.weight.device,
        )
    )
    r = random.Random(random_seed)
    indices = torch.tensor(r.sample(range(5000), prompt_length))
    with torch.no_grad():
        prompt_embedding.copy_(original_embedding(indices).unsqueeze(0))

    # find unique parameter name
    parameter_name = "prompt_embedding"
    parameter_name_index = 0
    while True:
        try:
            model.get_parameter(parameter_name)
        except AttributeError:
            break
        parameter_name_index += 1
        parameter_name = f"prompt_embedding_{parameter_name_index}"
    model.register_parameter(parameter_name, prompt_embedding)

    def patch_tensor(kwargs: Dict[str, torch.Tensor], key: str, value: Any = 0) -> None:
        t = kwargs.get(key)
        if t is None:
            return
        prefix = t.new_full((t.size(0), prompt_length) + t.shape[2:], value)
        kwargs[key] = torch.cat([prefix, t], dim=1)

    def patch_tensor_with_indices(
        kwargs: Dict[str, torch.Tensor], key: str, offset: int = 0
    ) -> None:
        t = kwargs.get(key)
        if t is None:
            return
        kwargs[key] = torch.cat(
            [
                torch.arange(0, prompt_length, dtype=t.dtype)
                .unsqueeze(0)
                .expand(t.size(0), prompt_length),
                t + offset,
            ],
            dim=1,
        )

    # Because PyTorch hooks don't support kwargs, we monkey patch the forward method 🙈
    old_forward = model.forward

    def new_forward(*args, **kwargs):

        # Massage the input to include the prompt
        if kwargs.get("past_key_values") is not None:
            # If we have already been running this model, we don't need to do anything with the prefix now.
            return old_forward(*args, **kwargs)

        inputs_embeds: Optional[torch.Tensor] = None
        input_ids = kwargs.pop("input_ids", None)
        if input_ids is not None:
            inputs_embeds = original_embedding(input_ids)

        inputs_embeds = kwargs.get("inputs_embeds", inputs_embeds)
        if inputs_embeds is not None:
            kwargs["inputs_embeds"] = torch.cat(
                [prompt_embedding.expand(inputs_embeds.size(0), -1, -1), inputs_embeds], dim=1
            )

        patch_tensor(kwargs, "labels")
        patch_tensor(kwargs, "attention_mask", 1)
        patch_tensor(kwargs, "token_type_ids")
        patch_tensor_with_indices(kwargs, "position_ids", prompt_length)

        # Run the model
        result = old_forward(*args, **kwargs)

        # Massage the output to look like the prompt was never there
        if isinstance(result, CausalLMOutputWithCrossAttentions):
            unpatch_tensor = lambda t: t[:, prompt_length:]  # noqa: E731
            if result.logits is not None:
                result.logits = unpatch_tensor(result.logits)
            if result.hidden_states is not None:
                result.hidden_states = tuple(map(unpatch_tensor, result.hidden_states))

            unpatch_attention_tensors = lambda t: t[:, :, prompt_length:]  # noqa: E731
            if result.attentions is not None:
                result.attentions = tuple(map(unpatch_attention_tensors, result.attentions))
            if result.cross_attentions is not None:
                result.cross_attentions = tuple(
                    map(unpatch_attention_tensors, result.cross_attentions)
                )

            return result
        else:
            logger.warning(
                "Unexpected result type from the transformer in soft_prompt_transformer: `%s`",
                result.__class__,
            )
            return result

    model.forward = new_forward  # type: ignore


Model.register("transformers::with_soft_prompt")(add_soft_prompt)  # type: ignore
