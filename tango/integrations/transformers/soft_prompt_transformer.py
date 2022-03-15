import random
from typing import Dict, Any
import logging

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from tango.integrations.torch import Model


logger = logging.getLogger(__name__)


class _WithPromptEmbedding(nn.Module):
    def __init__(
        self, original_embedding: nn.Embedding, prompt_length: int, random_seed: int = 1940
    ):
        super().__init__()

        self.prompt_length = prompt_length
        self.original_embedding = original_embedding
        self.prompt_embedding = nn.Embedding(prompt_length, self.original_embedding.embedding_dim)

        # following Lester et al. 2021 in initializing using the top 5000 random vocabs
        r = random.Random(random_seed)
        indices = torch.tensor(r.sample(range(5000), prompt_length))
        with torch.no_grad():
            self.prompt_embedding.weight.copy_(self.original_embedding(indices))

    def forward(self, input: torch.Tensor):
        embedded_prompt = self.prompt_embedding(input[:, : self.prompt_length])
        embedded_rest = self.original_embedding(input[:, self.prompt_length :])
        return torch.cat([embedded_prompt, embedded_rest], dim=1)

    @property
    def embedding_dim(self):
        return self.original_embedding.embedding_dim


def make_soft_prompt_transformer(model: Model, prompt_length: int) -> Model:
    """
    Takes a regular huggingface transformer, and equips it with a soft prompt.

    :param model: the original huggingface transformer. This model is augmented in-place!
    :param prompt_length: the length of the soft prompt, in tokens
    :return: the new transformer, equipped with the soft prompt.
    """
    assert isinstance(model, PreTrainedModel)

    model.set_input_embeddings(_WithPromptEmbedding(model.get_input_embeddings(), prompt_length))

    def patch_tensor(kwargs: Dict[str, torch.Tensor], key: str, value: Any = 0) -> None:
        try:
            t = kwargs[key]
        except KeyError:
            return
        prefix = t.new_full((t.size(0), prompt_length) + t.shape[2:], value)
        kwargs[key] = torch.cat([prefix, t], dim=1)

    def patch_tensor_with_indices(
        kwargs: Dict[str, torch.Tensor], key: str, offset: int = 0
    ) -> None:
        try:
            t = kwargs[key]
        except KeyError:
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
        if "past_key_values" in kwargs:
            # If we have already been running this model, we don't need to do anything with the prefix now.
            return old_forward(*args, **kwargs)
        patch_tensor_with_indices(kwargs, "input_ids")
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

    model.forward = new_forward
    return model


Model.register("transformers::with_soft_prompt")(make_soft_prompt_transformer)
