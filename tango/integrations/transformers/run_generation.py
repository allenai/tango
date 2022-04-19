import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Union, cast

import more_itertools
import torch
from datasets import Dataset
from datasets import DatasetDict as HfDatasetDict
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
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)

from tango import Format, JsonFormat, SqliteDictFormat, Step
from tango.common import DatasetDict
from tango.common.sequences import MappedSequence, SqliteSparseSequence
from tango.common.tqdm import Tqdm
from tango.integrations.torch import Model
from tango.integrations.torch.util import resolve_device, set_seed_all

logger = logging.getLogger(__name__)

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

SEQ2SEQ = AutoModelForSeq2SeqLM._model_mapping.keys()  # type: ignore
CAUSAL = AutoModelForCausalLM._model_mapping.keys()  # type: ignore


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


def _generate(
    model: Model,
    # TODO: Change type to `Tokenizer` once HF includes `convert_tokens_to_ids` in `PretrainedTokenizerBase` class.
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    prompts: Iterable[str],
    *,
    batch_size: int = 4,
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

    if not isinstance(model.config, tuple(SEQ2SEQ + CAUSAL)):
        raise NotImplementedError(
            "This function is only defined for huggingface models seq2seq/causal models."
        )

    device = resolve_device()
    set_seed_all(seed)

    tokenizer_kwargs: Dict[str, Any] = {}
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "[EOS]"})

    eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    # Seq2Seq models don't return their own prefix.
    seq2seq_model = model.config_class in SEQ2SEQ

    # HF does not do this? WTF?
    model.eval()

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
        result = {key: tensor.to(device) for key, tensor in result.items()}
        return result

    def prepare_batch_with_prefix(prompts: List[str]) -> Dict[str, torch.Tensor]:
        if len(prefix) > 0:
            prompts = [f"{prefix} {t}" for t in prompts]
        return prepare_batch_without_prefix(prompts)

    prepare_batch_fn = prepare_batch_with_prefix
    num_prefix_tokens: Optional[int] = None

    # transformer model-specific exceptions
    if isinstance(model, PreTrainedModel) and model.config_class:
        if model.config_class.model_type == "xlm":
            use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
            if hasattr(model.config, "lang2id") and use_lang_emb:
                model.config.lang_id = xlm_language
            # Original HF code ignores the prefix, but it looks like a bug?
            prepare_batch_fn = prepare_batch_without_prefix
            num_prefix_tokens = 0
        elif model.config_class.model_type in {"xlnet", "transfo-xl"}:
            prefix = prefix if prefix else PREFIX
        if model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
            # This actually doesn't work in the current version of transformers, which is probably a bug in the
            # transformers library.
            tokenizer_kwargs = {"add_space_before_punct_symbol": True}

    if num_prefix_tokens is None:
        num_prefix_tokens = len(tokenizer.tokenize(prefix))

    batches = more_itertools.chunked(Tqdm.tqdm(prompts, desc="Pre-processing prompts"), batch_size)
    encoded_batches = map(prepare_batch_fn, batches)

    for encoded_batch in Tqdm.tqdm(encoded_batches, desc="Processing batches"):
        if seq2seq_model:
            length = max_length
        else:
            length = adjust_length_to_model(max_length + encoded_batch["input_ids"].size(1), model)
        with torch.inference_mode():
            generated_sequences: torch.Tensor = model.generate(  # type: ignore
                **encoded_batch,
                max_length=length,
                temperature=temperature,
                top_k=k,
                top_p=p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                synced_gpus=False,  # Needs to be True if we have more than one GPU running.
            )

        generated_sequences = generated_sequences.view(
            -1, num_return_sequences, *generated_sequences.shape[1:]
        ).to("cpu")

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
        generated_sequences_list = [
            [strip_special_tokens(sequence) for sequence in per_prompt_sequences]
            for per_prompt_sequences in generated_sequences
        ]

        # strip prefix
        if not seq2seq_model:
            generated_sequences_list = [
                [sequence[num_prefix_tokens:] for sequence in per_prompt_sequences]
                for per_prompt_sequences in generated_sequences_list
            ]

        texts = [
            tokenizer.batch_decode(per_prompt_sequences, clean_up_tokenization_spaces=True)
            for per_prompt_sequences in generated_sequences_list
        ]

        yield from texts


def _generate_with_model_name(model_name: str, *args, **kwargs) -> Iterable[List[str]]:
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _generate(model, tokenizer, *args, **kwargs)


@Step.register("transformers::run_generation")
class RunGeneration(Step[Iterable[List[str]]]):
    """
    A step that runs seq2seq Huggingface models in inference mode.

    .. tip::
        Registered as a :class:`~tango.step.Step` under the name "transformers::run_generation".
    """

    FORMAT: Format = JsonFormat("gz")
    VERSION = "001"
    SKIP_ID_ARGUMENTS = {"batch_size"}

    # TODO: multiple GPUs

    def run(  # type: ignore
        self,
        model: Union[str, Model],
        prompts: Iterable[str],
        *,
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
        batch_size: int = 4,
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
        """
        Run a Huggingface seq2seq model in inference mode.

        :param model:
            The name of the model to run. Any name that works in the transformers library works here.
            Or, you can directly provide the model to run.
        :param prompts:
            The prompts to run through the model. You can specify prompts directly in the config, but
            more commonly the prompts are produced by another step that reads a dataset, for example.
        :param tokenizer:
            The tokenizer to run.
        :param batch_size:
            The number of sequences to process at one time. This has no bearing on the output, so
            you can change this number without invalidating cached results.
        :param max_length:
            The maximum number of tokens/word pieces that the model will generate. For models that
            extend the prompt, the prefix does not count towards this limit.
        :param temperature:
            Passed directly to transformer's ``generate()`` method.
            The value used to model the next token probabilities.
        :param repetition_penalty:
            Passed directly to transformer's ``generate()`` method.
            The parameter for repetition penalty. 1.0 means no penalty.
        :param k:
            Passed directly to transformer's ``generate()`` method.
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        :param p:
            Passed directly to transformer's ``generate()`` method.
            If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher
            are kept for generation.
        :param prefix:
            A prefix that gets pre-pended to all prompts.
        :param xlm_language:
            For the XLM model, this is a way to specify the language you want to use.
        :param seed:
            Random seed
        :param num_return_sequences:
            The number of generations to return for each prompt.
        :param fp16:
            Whether to use 16-bit floats.

        :returns:
            Returns an iterator of lists of string. Each list contains the predictions for one prompt.
        """
        if isinstance(model, str):
            try:
                model = cast(Model, AutoModelForSeq2SeqLM.from_pretrained(model))
            except ValueError:
                model = cast(Model, AutoModelForCausalLM.from_pretrained(model))

        tokenizer = tokenizer or AutoTokenizer.from_pretrained(model.name_or_path)

        return _generate(
            model,
            tokenizer,
            prompts,
            batch_size=batch_size,
            max_length=max_length,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            k=k,
            p=p,
            prefix=prefix,
            xlm_language=xlm_language,
            seed=seed,
            num_return_sequences=num_return_sequences,
            fp16=fp16,
        )


@Step.register("transformers::run_generation_dataset")
class RunGenerationDataset(Step[DatasetDict]):
    """
    A step that runs seq2seq Huggingface models in inference mode.

    This is similar to :class:`RunGeneration`, but it takes a dataset as input and produces
    a new dataset as output, which contains the predictions in a new field.

    .. tip::
        Registered as a :class:`~tango.step.Step` under the name "transformers::run_generation_dataset".
    """

    FORMAT: Format = SqliteDictFormat()
    VERSION = "002"
    SKIP_ID_ARGUMENTS = {"batch_size"}

    def run(  # type: ignore
        self,
        model: Union[str, Model],
        input: Union[DatasetDict, HfDatasetDict],
        prompt_field: str,
        *,
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
        output_field: Optional[str] = None,
        splits: Optional[Union[str, Set[str]]] = None,
        batch_size: int = 4,
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
    ) -> DatasetDict:
        """
        Augment an input dataset with generations from a Huggingface seq2seq model.

        :param model:
            The name of the model to run. Any name that works in the transformers library works here.
            Or, you can directly provide the model to run.
        :param input:
            The input dataset.
        :param prompt_field:
            The field in the dataset that contains the text of the prompts.
        :param tokenizer:
            The tokenizer to run.
        :param output_field:
            The field in the dataset that we will write the predictions into. In the result, this field
            will contain ``List[str]``.
        :param splits:
            A split, or set of splits, to process. If this is not specified, we will process all splits.
        :param batch_size:
            The number of sequences to process at one time. This has no bearing on the output, so
            you can change this number without invalidating cached results.
        :param max_length:
            The maximum number of tokens/word pieces that the model will generate. For models that
            extend the prompt, the prefix does not count towards this limit.
        :param temperature:
            Passed directly to transformer's `generate()` method.
            The value used to model the next token probabilities.
        :param repetition_penalty:
            Passed directly to transformer's `generate()` method.
            The parameter for repetition penalty. 1.0 means no penalty.
        :param k:
            Passed directly to transformer's `generate()` method.
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        :param p:
            Passed directly to transformer's `generate()` method.
            If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher
            are kept for generation.
        :param prefix:
            A prefix that gets pre-pended to all prompts.
        :param xlm_language:
            For the XLM model, this is a way to specify the language you want to use.
        :param seed:
            Random seed
        :param num_return_sequences:
            The number of generations to return for each prompt.
        :param fp16:
            Whether to use 16-bit floats.

        :returns:
            Returns a dataset with an extra field containing the predictions.
        """

        if isinstance(model, str):
            try:
                model = cast(Model, AutoModelForSeq2SeqLM.from_pretrained(model))
            except ValueError:
                model = cast(Model, AutoModelForCausalLM.from_pretrained(model))

        tokenizer = tokenizer or AutoTokenizer.from_pretrained(model.name_or_path)

        if isinstance(input, HfDatasetDict):
            input = DatasetDict(input, {})
        if splits is None:
            splits = input.keys()
        elif isinstance(splits, str):
            splits = {splits}

        result: Dict[str, Sequence] = {}
        for split_name, input_split in input.items():
            if split_name in splits:
                output_split = SqliteSparseSequence(self.work_dir / f"{split_name}.sqlite")
                if len(output_split) > 0:
                    logger.info(
                        "Found %d items already generated. Will generate %d more.",
                        len(output_split),
                        len(input_split) - len(output_split),
                    )
                if len(output_split) > 0:
                    if isinstance(input_split, Dataset):
                        input_split = input_split.select(range(len(output_split), len(input_split)))
                    else:
                        input_split = input_split[len(output_split) :]
                prompts = MappedSequence(lambda i: i[prompt_field], input_split)
                generations = _generate(
                    model,
                    tokenizer,
                    prompts,
                    batch_size=batch_size,
                    max_length=max_length,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    k=k,
                    p=p,
                    prefix=prefix,
                    xlm_language=xlm_language,
                    seed=seed,
                    num_return_sequences=num_return_sequences,
                    fp16=fp16,
                )
                for instance, generation in zip(input_split, generations):
                    output_split.append(
                        {**instance, **{output_field or prompt_field + "_generated": generation}}
                    )
                result[split_name] = output_split
            else:
                result[split_name] = input_split

        return DatasetDict(result, input.metadata)
