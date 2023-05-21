import os

import accelerate
from langchain import LLMChain, PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

from wizardlm_langchain.helpers import (
    AttributeDict,
    convert_to_bytes,
    get_available_memory,
)
from wizardlm_langchain.model import load_quantized_model

GPTQ_MODEL_DIR = os.getenv("MODEL_DIR", "./models/")
MODEL_NAME = os.getenv("MODEL_NAME", "wizardLM-7B-GPTQ")
CPU_MEM_BUFFER = os.getenv("CPU_MEM_BUFFER", "3GiB")


def create_gptq_llm_chain(cpu_mem_buffer):
    args = {
        "wbits": 4,
        "groupsize": 128,
        "model_type": "llama",
        "model_dir": GPTQ_MODEL_DIR,
    }

    model, tokenizer = load_quantized_model(MODEL_NAME, args=AttributeDict(args))
    cpu_mem, gpu_mem_map = get_available_memory(cpu_mem_buffer)
    print(f"Detected Memory: System={cpu_mem}, GPU(s)={gpu_mem_map}")

    max_memory = {**gpu_mem_map, "cpu": cpu_mem}

    device_map = accelerate.infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=["LlamaDecoderLayer"]
    )

    model = accelerate.dispatch_model(
        model, device_map=device_map, offload_buffers=True
    )

    print(f"Memory footprint of model: {model.get_memory_footprint() / (1024 * 1024)}")

    template = """Question: {question}

    Answer: Let's think step by step."""

    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        device_map=device_map,
    )

    local_llm = HuggingFacePipeline(pipeline=llm_pipeline)

    prompt = PromptTemplate(template=template, input_variables=["question"])

    llm_chain = LLMChain(prompt=prompt, llm=local_llm)

    return llm_chain


def ask_ai_interactive(llm_chain):
    chat_history = []
    print(
        "\n\nWelcome to the GPTQ Chat Interface! Type 'exit' to stop when you're done."
    )
    while True:
        query = input("\nType your question here: ")

        if query.lower() == "exit":
            break

        result = llm_chain({"question": query, "chat_history": chat_history})
        chat_history.append((query, result["text"]))

        print("Answer:", result["text"])


if __name__ == "__main__":
    mem_buf = convert_to_bytes(CPU_MEM_BUFFER)
    llm_chain = create_gptq_llm_chain(mem_buf)
    ask_ai_interactive(llm_chain)
