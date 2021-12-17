import logging
from typing import Any, Dict, Iterable, List, Optional

import more_itertools
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)

from tango import Format, JsonFormat, Step
from tango.common.logging import make_tqdm
from tango.common.util import threaded_generator

logger = logging.getLogger(__name__)
tqdm = make_tqdm(logger)

#
# A lot of the code in this step is stolen from the run_generation.py script in transformers. Unfortunately their
# examples don't ship when you `pip install transformers`, so we have to duplicate it here.
#

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PREFIX = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def adjust_length_to_model(length, model):
    max_sequence_length = (
        model.config.max_position_embeddings
        if hasattr(model.config, "max_position_embeddings")
        else MAX_LENGTH
    )
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


@Step.register("transformers::run_generation")
class RunGeneration(Step[Iterable[List[str]]]):
    FORMAT: Format = JsonFormat("gz")
    VERSION = "001"

    def run(  # type: ignore
        self,
        prompts: Iterable[str],
        model_name: str,
        *,
        batch_size: int = 4,  # TODO: This should not be part of the unique id
        max_length: int = 20,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        k: int = 0,
        p: float = 0.9,
        prefix: str = "",
        xlm_language: str = "",
        seed: int = 42,
        num_return_sequences: int = 1,
        fp16: bool = False,
    ) -> Iterable[List[str]]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()

        logger.info(f"device: {device}, n_gpu: {n_gpu}, 16-bits: {fp16}")

        np.random.seed(seed)
        torch.manual_seed(seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer_kwargs: Dict[str, Any] = {}
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            seq2seq_model = True   # Seq2Seq models don't return their own prefix.
        except ValueError:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            seq2seq_model = False

        # HF does not do this? WTF?
        model.eval()
        # Memory saving hack according to Iz
        for parameter in model.parameters():
            parameter.required_grad = False

        model.to(device)
        if fp16:
            model.half()

        def prepare_batch_without_prefix(prompts: List[str]) -> Dict[str, torch.Tensor]:
            result = tokenizer.batch_encode_plus(
                prompts,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
                **tokenizer_kwargs,
            )
            result = {
                key: tensor.to(device)
                for key, tensor in result.items()
            }
            return result

        def prepare_batch_with_prefix(prompts: List[str]) -> Dict[str, torch.Tensor]:
            if len(prefix) > 0:
                prompts = [f"{prefix} {t}" for t in prompts]
            return prepare_batch_without_prefix(prompts)

        prepare_batch_fn = prepare_batch_with_prefix
        num_prefix_tokens: Optional[int] = None

        # model-specific exceptions
        if model.config_class.model_type == "ctrl":
            if temperature > 0.7:
                logger.warning(
                    "CTRL typically works better with lower temperatures (and lower top_k)."
                )

            def prepare_batch_fn(prompts: List[str]) -> Dict[str, torch.Tensor]:
                encoded_prompts = prepare_batch_without_prefix([prefix + t for t in prompts])
                if not any(
                    encoded_prompts["token_ids"][0, 0] == x
                    for x in tokenizer.control_codes.values()
                ):
                    logger.warning(
                        "You are not starting your generation from a control code "
                        "so you won't get good results!"
                    )
                return encoded_prompts

        elif model.config_class.model_type == "xlm":
            use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
            if hasattr(model.config, "lang2id") and use_lang_emb:
                model.config.lang_id = xlm_language
            # Original HF code ignores the prefix, but it looks like a bug?
            prepare_batch_fn = prepare_batch_without_prefix
            num_prefix_tokens = 0
        elif model.config_class.model_type in {"xlnet", "transfo-xl"}:
            prefix = prefix if prefix else PREFIX
        if model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
            tokenizer_kwargs = {"add_space_before_punct_symbol": True}

        if num_prefix_tokens is None:
            num_prefix_tokens = len(tokenizer.tokenize(prefix))

        batches = more_itertools.chunked(tqdm(prompts, desc="Pre-processing prompts"), batch_size)
        encoded_batches = map(prepare_batch_fn, batches)
        encoded_batches = threaded_generator(encoded_batches)

        for encoded_batch in tqdm(encoded_batches, desc="Processing batches"):
            if seq2seq_model:
                length = max_length
            else:
                length = adjust_length_to_model(
                    max_length + encoded_batch["input_ids"].size(1), model
                )
            generated_sequences = model.generate(
                **encoded_batch,
                max_length=length,
                temperature=temperature,
                top_k=k,
                top_p=p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                synced_gpus=n_gpu > 1,
            )

            # TODO: Run on GPU
            # TODO: Run on multiple GPUs

            generated_sequences = generated_sequences.view(
                -1, num_return_sequences, *generated_sequences.shape[1:]
            ).to("cpu")

            # strip prefix tokens
            if not seq2seq_model:
                generated_sequences = generated_sequences[..., num_prefix_tokens:]

            def strip_special_tokens(t: torch.Tensor) -> torch.Tensor:
                # amazing that torch has no capability for this
                start = 0
                while start < len(t) and int(t[start]) in {0, eos_token_id, pad_token_id}:
                    start += 1
                end = len(t)
                while int(t[end - 1]) in {0, eos_token_id, pad_token_id} and end > start:
                    end -= 1
                return t[start:end]

            # strip padding
            generated_sequences = [
                [
                    strip_special_tokens(sequence)
                    for sequence in per_prompt_sequences
                ]
                for per_prompt_sequences in generated_sequences
            ]

            texts = [
                tokenizer.batch_decode(per_prompt_sequences, clean_up_tokenization_spaces=True)
                for per_prompt_sequences in generated_sequences
            ]

            yield from texts
