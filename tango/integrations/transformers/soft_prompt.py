import inspect
import logging
import random
from typing import Any, Dict, Optional

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
    Seq2SeqModelOutput,
)

from tango.integrations.torch import Model

logger = logging.getLogger(__name__)


def _get_bound_args_with_decorators(fn, *args, **kwargs):
    while True:
        try:
            fn = fn.__wrapped__
        except AttributeError:
            break
    signature = inspect.Signature.from_callable(fn)
    return signature.bind(*args, **kwargs)


def add_soft_prompt(
    model: Model,
    prompt_length: int,
    *,
    only_prompt_is_trainable: bool = True,
    initialize_from_top_embeddings: Optional[int] = 5000,
    random_seed: int = 1940,
) -> None:
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
    :param only_prompt_is_trainable: freezes the original model's weights, leaving only the prompt trainable
    :param initialize_from_top_embeddings: Prompt embeddings are initialized from a random selection of the top n
                                           word piece embeddings from the original model. This is how you set n.
    :param random_seed: random seed used to initialize the prompt embeddings

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
    if initialize_from_top_embeddings is None:
        initialize_from_top_embeddings = original_embedding.num_embeddings
    indices = torch.tensor(r.sample(range(initialize_from_top_embeddings), prompt_length))
    with torch.no_grad():
        prompt_embedding.copy_(original_embedding(indices).unsqueeze(0))

    if only_prompt_is_trainable:
        for param in model.parameters():
            param.requires_grad = False

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

    old_forward = model.forward

    def new_forward(*args, **kwargs):
        # Massage the input to include the prompt
        if kwargs.get("past_key_values") is not None:
            # If we have already been running this model, we don't need to do anything with the prefix now.
            return old_forward(*args, **kwargs)
        if kwargs.get("encoder_outputs") is not None:
            # For encoder/decoder models, this runs only on the encoder. If we already have encoder outputs,
            # we don't have to do anything.
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
        unpatch_tensor = lambda t: t[:, prompt_length:]  # noqa: E731
        unpatch_attention_tensor = lambda t: t[:, :, prompt_length:]  # noqa: E731
        unpatch_kv_tensor = unpatch_attention_tensor
        if isinstance(result, CausalLMOutputWithCrossAttentions):
            if result.logits is not None:
                result.logits = unpatch_tensor(result.logits)
            if result.hidden_states is not None:
                result.hidden_states = tuple(map(unpatch_tensor, result.hidden_states))
            if result.attentions is not None:
                result.attentions = tuple(map(unpatch_attention_tensor, result.attentions))
            if result.cross_attentions is not None:
                result.cross_attentions = tuple(
                    map(unpatch_attention_tensor, result.cross_attentions)
                )
            return result
        elif isinstance(result, Seq2SeqModelOutput):
            if result.last_hidden_state is not None:
                result.last_hidden_state = unpatch_tensor(result.last_hidden_state)
            if result.past_key_values is not None:
                result.past_key_values = tuple(map(unpatch_kv_tensor, result.past_key_values))
            if result.encoder_hidden_states is not None:
                result.hidden_states = tuple(map(unpatch_tensor, result.hidden_states))
            if result.encoder_attentions is not None:
                result.attentions = tuple(map(unpatch_attention_tensor, result.attentions))
            if result.cross_attentions is not None:
                result.cross_attentions = tuple(
                    map(unpatch_attention_tensor, result.cross_attentions)
                )
            return result
        else:
            logger.warning(
                "Unexpected result type from the transformer in soft_prompt_transformer: `%s`",
                result.__class__,
            )
            return result

    model.forward = new_forward  # type: ignore

    # For encoder/decoder models, HF doesn't call `forward()` like it should when you use `generate()`. Instead, it
    # calls the encoder separately, and then passes the results into `forward()`. So in that case, we have to patch
    # this too.
    if model.config.is_encoder_decoder:
        old_generate = model.generate

        def new_generate(*args, **kwargs):
            args = (model,) + args
            ba = _get_bound_args_with_decorators(old_generate, *args, **kwargs)
            del ba.arguments["self"]

            if "encoder_outputs" in ba.arguments:
                # For encoder/decoder models, this runs only on the encoder. If we already have encoder outputs,
                # we don't have to do anything.
                return old_generate(*ba.args, **ba.kwargs)

            inputs_embeds: Optional[torch.Tensor] = None
            inputs = ba.arguments.pop("inputs", None)
            if inputs is not None:
                inputs_embeds = original_embedding(inputs)

            inputs_embeds = ba.arguments.pop("inputs_embeds", inputs_embeds)
            if inputs_embeds is not None:
                inputs_embeds = torch.cat(
                    [prompt_embedding.expand(inputs_embeds.size(0), -1, -1), inputs_embeds], dim=1
                )

            assert callable(model.get_encoder)
            encoder = model.get_encoder()
            kwargs = ba.kwargs
            kwargs["encoder_outputs"] = encoder(inputs_embeds=inputs_embeds, return_dict=True)

            return old_generate(*ba.args, **kwargs)

        model.generate = new_generate  # type: ignore


def _with_soft_prompt(
    model: Model,
    prompt_length: int,
    *,
    only_prompt_is_trainable: bool = True,
    initialize_from_top_embeddings: Optional[int] = 5000,
    random_seed: int = 1940,
) -> Model:
    """To initialize a soft-prompt model as a Registrable (i.e., to use it from a config file), we need a variant
    of this function that returns the resulting model. This is that variant."""
    add_soft_prompt(
        model,
        prompt_length,
        only_prompt_is_trainable=only_prompt_is_trainable,
        initialize_from_top_embeddings=initialize_from_top_embeddings,
        random_seed=random_seed,
    )
    return model


Model.register("transformers::with_soft_prompt")(_with_soft_prompt)  # type: ignore
