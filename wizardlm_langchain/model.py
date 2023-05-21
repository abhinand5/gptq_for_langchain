import inspect
import logging
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaTokenizer

from .helpers import find_quantized_model_file

try:
    from gptq_for_llama.modelutils import find_layers
    from gptq_for_llama.quant import make_quant

    is_triton = False
except ImportError:
    sys.path.insert(0, str(Path("../.tmp")))
    from gptq_for_llama import quant
    from gptq_for_llama.utils import find_layers

    is_triton = True


# Inspired from https://github.com/oobabooga/text-generation-webui
def _load_quant(
    model,
    checkpoint,
    wbits,
    groupsize=-1,
    faster_kernel=False,
    eval=False,
    exclude_layers=["lm_head"],
    kernel_switch_threshold=128,
):
    def noop(*args, **kwargs):
        pass

    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    logging.info(f"Model Config: {config}")

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    torch.set_default_dtype(torch.float)

    if eval:
        model = model.eval()

    layers = find_layers(model)
    for name in exclude_layers:
        if name in layers:
            del layers[name]

    if not is_triton:
        gptq_args = inspect.getfullargspec(make_quant).args
        make_quant_kwargs = {
            "module": model,
            "names": layers,
            "bits": wbits,
        }
        if "groupsize" in gptq_args:
            make_quant_kwargs["groupsize"] = groupsize
        if "faster" in gptq_args:
            make_quant_kwargs["faster"] = faster_kernel
        if "kernel_switch_threshold" in gptq_args:
            make_quant_kwargs["kernel_switch_threshold"] = kernel_switch_threshold

        make_quant(**make_quant_kwargs)
    else:
        logging.exception("Triton not supported!")

    del layers

    if checkpoint.endswith(".safetensors"):
        from safetensors.torch import load_file as safe_load

        model.load_state_dict(safe_load(checkpoint), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)

    model.seqlen = 2048
    return model


# Inspired from https://github.com/oobabooga/text-generation-webui
def load_quantized_model(model_name, args, load_tokenizer=True):
    tokenizer = None
    path_to_model = Path(f"{args.model_dir}/{model_name}")
    pt_path = find_quantized_model_file(model_name, args)
    if not pt_path:
        print(pt_path)
        logging.error(
            "Could not find the quantized model in .pt or .safetensors format, exiting..."
        )
        return
    else:
        logging.info(f"Found the following quantized model: {pt_path}")

    threshold = args.threshold if args.threshold else 128

    model = _load_quant(
        str(path_to_model),
        str(pt_path),
        args.wbits,
        args.groupsize,
        kernel_switch_threshold=threshold,
    )

    model = model.to(torch.device("cuda:0"))

    if load_tokenizer:
        tokenizer = LlamaTokenizer.from_pretrained(
            Path(f"{args.model_dir}/{model_name}/"), clean_up_tokenization_spaces=True
        )

        try:
            tokenizer.eos_token_id = 2
            tokenizer.bos_token_id = 1
            tokenizer.pad_token_id = 0
        except:
            pass

    return model, tokenizer
