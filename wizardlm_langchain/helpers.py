import logging
from pathlib import Path

import psutil
import torch


# Based on https://github.com/oobabooga/text-generation-webui/blob/main/modules/GPTQ_loader.py
def find_quantized_model_file(model_name, args):
    if args.checkpoint:
        return Path(args.checkpoint)

    path_to_model = Path(f"{args.model_dir}/{model_name}")
    print(f"Path to Model: {path_to_model}")
    pt_path = None
    priority_name_list = [
        Path(f"{args.model_dir}/{model_name}{hyphen}{args.wbits}bit{group}{ext}")
        for group in ([f"-{args.groupsize}g", ""] if args.groupsize > 0 else [""])
        for ext in [".safetensors", ".pt"]
        for hyphen in ["-", f"/{model_name}-", "/"]
    ]
    for path in priority_name_list:
        if path.exists():
            pt_path = path
            break

    # If the model hasn't been found with a well-behaved name, pick the last .pt
    # or the last .safetensors found in its folder as a last resort
    if not pt_path:
        found_pts = list(path_to_model.glob("*.pt"))
        found_safetensors = list(path_to_model.glob("*.safetensors"))
        pt_path = None

        if len(found_pts) > 0:
            if len(found_pts) > 1:
                logging.warning(
                    "More than one .pt model has been found. The last one will be selected. It could be wrong."
                )

            pt_path = found_pts[-1]
        elif len(found_safetensors) > 0:
            if len(found_pts) > 1:
                logging.warning(
                    "More than one .safetensors model has been found. The last one will be selected. It could be wrong."
                )

            pt_path = found_safetensors[-1]

    return pt_path


def get_available_memory(system_memory_buffer):
    # Get system memory
    system_memory = psutil.virtual_memory().available

    # Calculate the maximum system memory based on the buffer
    max_system_memory = system_memory - system_memory_buffer

    # Limit system memory to the maximum allowed
    system_memory = min(system_memory, max_system_memory)

    # Get available GPU memory for all devices
    available_memory = {}
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for device_id in range(device_count):
            gpu_memory = torch.cuda.get_device_properties(device_id).total_memory
            available_memory[device_id] = gpu_memory

    return system_memory, available_memory


def convert_to_bytes(size_str):
    # Check if size_str ends with 'GiB' or 'MiB'
    if size_str.endswith("GiB"):
        size = float(size_str[:-3]) * 1024 * 1024 * 1024
    elif size_str.endswith("MiB"):
        size = float(size_str[:-3]) * 1024 * 1024
    else:
        raise ValueError("Invalid size format. Expected 'GiB' or 'MiB' suffix.")

    return int(size)


class AttributeDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
