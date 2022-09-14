import re
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_utils import Conv1D


@dataclass
class WithIA3Config:
    ia3_param_names: str
    attention_modules: str
    mlp_modules: str
    mlp_layers: str
    fused_qkv_layers: Optional[str] = None
    k_layers: Optional[str] = None
    v_layers: Optional[str] = None


GPT_J_IA3_CONFIG = WithIA3Config(
    attention_modules=".*attn",
    k_layers="k_proj",
    v_layers="v_proj",
    mlp_modules=".*mlp",
    mlp_layers="fc_in",
    ia3_param_names="ia3",
)

GPT_2_IA3_CONFIG = WithIA3Config(
    attention_modules=".*attn",
    fused_qkv_layers="c_attn",
    mlp_modules=".*mlp",
    mlp_layers="c_fc",
    ia3_param_names="ia3",
)

OPT_IA3_CONFIG = WithIA3Config(
    attention_modules=".*self_attn",
    k_layers="k_proj",
    v_layers="v_proj",
    mlp_modules=r".*layers\.\d*",
    mlp_layers="fc1",
    ia3_param_names="ia3",
)

BLOOM_IA3_CONFIG = WithIA3Config(
    attention_modules=".*self_attention",
    fused_qkv_layers="query_key_value",
    mlp_modules=".*mlp",
    mlp_layers="dense_h_to_4h",
    ia3_param_names="ia3",
)

MODEL_NAME_TO_CONFIG = {
    "sshleifer/tiny-gpt2": GPT_2_IA3_CONFIG,
    "gpt2": GPT_2_IA3_CONFIG,
    "gpt2-medium": GPT_2_IA3_CONFIG,
    "gpt2-large": GPT_2_IA3_CONFIG,
    "gpt2-xl": GPT_2_IA3_CONFIG,
    "bigscience/bloom-560m": BLOOM_IA3_CONFIG,
    "bigscience/bloom-1b1": BLOOM_IA3_CONFIG,
    "bigscience/bloom-1b7": BLOOM_IA3_CONFIG,
    "bigscience/bloom-3b": BLOOM_IA3_CONFIG,
    "bigscience/bloom-7b1": BLOOM_IA3_CONFIG,
    "bigscience/bloom": BLOOM_IA3_CONFIG,
    "facebook/opt-125m": OPT_IA3_CONFIG,
    "facebook/opt-350m": OPT_IA3_CONFIG,
    "facebook/opt-1.3b": OPT_IA3_CONFIG,
    "facebook/opt-2.7b": OPT_IA3_CONFIG,
    "facebook/opt-6.7b": OPT_IA3_CONFIG,
    "facebook/opt-13b": OPT_IA3_CONFIG,
    "facebook/opt-30b": OPT_IA3_CONFIG,
    "facebook/opt-66b": OPT_IA3_CONFIG,
    "EleutherAI/gpt-j-6B": GPT_J_IA3_CONFIG,
}


class LinearWithIA3(nn.Module):
    def __init__(self, linear_layer, ia3_param_names, unfuse_size: int = None):
        super().__init__()

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.unfuse_size = unfuse_size

        self.weight = linear_layer.weight
        self.bias = linear_layer.bias

        self.ia3_param_names = ia3_param_names

        # if (q,k,v) are stacked into one layer
        if unfuse_size is not None:
            assert linear_layer.out_features == unfuse_size * 3
            # IA3 only operates on k and v (not q), thus the "* 2"
            setattr(self, ia3_param_names, nn.Parameter(torch.ones(unfuse_size * 2, 1)))
        else:
            setattr(self, ia3_param_names, nn.Parameter(torch.ones(self.out_features, 1)))

    def forward(self, x):
        x = F.linear(x, self.weight, self.bias)

        ia3_params = getattr(self, self.ia3_param_names)

        if ia3_params.requires_grad:
            if self.unfuse_size is not None:
                # non_q means k and v
                q, non_q = x[:, :, : self.unfuse_size], x[:, :, self.unfuse_size :]
                ia3_params = getattr(self, self.ia3_param_names)
                non_q = non_q * ia3_params.flatten()
                x = torch.cat([q, non_q], dim=2)
            else:
                x = x * ia3_params.flatten()

        return x


class Conv1DWithIA3(nn.Module):
    def __init__(self, conv1d_layer, ia3_param_names, unfuse_size: int = None):
        super().__init__()

        # nf: number of output features; nx: number of input features
        self.nf = conv1d_layer.nf
        self.unfuse_size = unfuse_size

        self.weight = conv1d_layer.weight
        self.bias = conv1d_layer.bias

        self.ia3_param_names = ia3_param_names

        # in c_att parameters, (q,k,v) linear layers are stacked into one Conv1D layer
        if unfuse_size is not None:
            assert conv1d_layer.nf == unfuse_size * 3
            # but IA3 only operates on k and v (not q), thus the "* 2"
            setattr(self, ia3_param_names, nn.Parameter(torch.ones(unfuse_size * 2, 1)))
        else:
            setattr(self, ia3_param_names, nn.Parameter(torch.ones(self.nf, 1)))

    def forward(self, x):
        # copied and pasted from the original Conv1D implemnetation
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)  # ... * self.nf

        ia3_params = getattr(self, self.ia3_param_names)

        if ia3_params.requires_grad:
            if self.unfuse_size is not None:
                # non_q means k and v
                q, non_q = x[:, :, : self.unfuse_size], x[:, :, self.unfuse_size :]
                ia3_params = getattr(self, self.ia3_param_names)
                non_q = non_q * ia3_params.flatten()
                x = torch.cat([q, non_q], dim=2)
            else:
                x = x * ia3_params.flatten()

        return x


def modify_with_ia3(
    transformer: PreTrainedModel, config: WithIA3Config, *, only_ia3_requires_grad: bool = True
) -> PreTrainedModel:
    """
    A function to add ia3 adaptors to the given transformer. Code modified from
    `t-few <https://github.com/r-three/t-few/blob/217cfa3b73aa66a07594826e4ebbbc516b331461/src/models/lora.py>`_
    and Qinyuan Ye


    :param model:
        A :class:`~transformers.PreTrainedModel` to modify.
    :param config:
        A :class:`~tango.integrations.transformers.ia3.WithIA3Config` that specifies the layers to modify.
    :param only_ia3_requires_grad:
        A `bool`, `True` if `requires_grad` should only be set on ia3 paramenters in the output model.

    Examples
    --------

    You can use the provided configurations:

    .. testcode::

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from tango.integrations.transformers.ia3 import modify_with_ia3, GPT_2_IA3_CONFIG

        model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        model = modify_with_ia3(model, GPT_2_IA3_CONFIG)

    Or you can write your own configuration with regex matching the layers to modify and their parents:

    .. testcode::

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from tango.integrations.transformers.ia3 import modify_with_ia3

        my_config = WithIA3Config(
            attention_modules=".*attn",
            fused_qkv_layers="c_attn",
            mlp_modules=".*mlp",
            mlp_layers="c_fc",
            ia3_param_names="ia3",
        )

        model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        model = modify_with_ia3(model, my_config)
    """
    for m_name, module in dict(transformer.named_modules()).items():  # type: ignore
        if re.fullmatch(config.attention_modules, m_name) or re.fullmatch(
            config.mlp_modules, m_name
        ):
            attn_layers = [
                regex
                for regex in (config.fused_qkv_layers, config.k_layers, config.v_layers)
                if regex is not None
            ]
            layers_to_change = (
                "|".join(attn_layers)
                if re.fullmatch(config.attention_modules, m_name)
                else config.mlp_layers
            )
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(layers_to_change, c_name):
                    assert isinstance(layer, Conv1D) or isinstance(
                        layer, nn.Linear
                    ), "This code only supports Conv1D and nn.Linear"
                    adaptor_class = Conv1DWithIA3 if isinstance(layer, Conv1D) else LinearWithIA3
                    new_module = adaptor_class(
                        layer,
                        config.ia3_param_names,
                        unfuse_size=transformer.config.hidden_size  # type: ignore
                        if config.fused_qkv_layers and re.fullmatch(config.fused_qkv_layers, c_name)
                        else None,
                    )
                    setattr(module, c_name, new_module)

    if only_ia3_requires_grad:
        transformer.requires_grad_(False)  # type: ignore
        for p_name, v in dict(transformer.named_parameters()).items():  # type: ignore
            if re.fullmatch(".*" + config.ia3_param_names + ".*", p_name):
                v.requires_grad_(True)

    return transformer
