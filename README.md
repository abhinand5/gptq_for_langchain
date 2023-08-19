# GPTQ models in LangChain

## Intro
This repository provides a potential framework with practical examples for developing applications powered by quantized open-source Language Model Models (LLMs) in conjunction with LangChain. Specifically, this guide focuses on the implementation and utilization of 4-bit Quantized GPTQ variants of various LLMs, such as WizardLM and WizardLM-Mega. While resources dedicated to this specific topic are limited online, this repository aims to bridge that gap and offer comprehensive guides.

Running LLMs locally offers numerous advantages, with privacy being a key factor. By keeping your data within your own hardware, you can leverage the capabilities of these models without relying on external APIs, ensuring greater control over your data and enabling the development of exciting applications.

## Prerequisites

- Ubuntu 22.04 / WSL2 Ubuntu for Windows
- Nvidia GPU with at least 6GB VRAM 
    - 6GB VRAM is enough for loading 4-bit 7B models
- 8 CPU threads
- 16 GB RAM is recommended
- Nvidia drivers already installed (`nvidia-smi` command should work)
- Suffucient disk space for packages, drivers and model files (~30GB, might vary depending on system)
- Git LFS

> **Note:** Docker support will be added.

## Tested Models
The following GPTQ models are supported for now:

- [wizardLM-7B-GPTQ](https://huggingface.co/TheBloke/wizardLM-7B-GPTQ)
- [WizardLM-7B-uncensored-GPTQ](https://huggingface.co/TheBloke/WizardLM-7B-uncensored-GPTQ)

## Getting up and running

In order to start using GPTQ models with langchain, there are a few important steps:

1. Set up Python Environment
2. Install the right versions of Pytorch and CUDA toolkit
3. Correctly set up `quant_cuda`
4. Download the GPTQ models from HuggingFace
5. After the above steps you can run `demo.py` and use the LLM with LangChain just like how you do it for OpenAI models.

### Creating the conda environment

Install Miniconda by following the instructions from the [official site](https://docs.conda.io/en/latest/miniconda.html).

To check whether conda was set up correctly

`$ conda --version`

Initialize conda on your shell

`$ conda init`

Create a new conda environment, make sure to use the specified Python version because it was tested only on `3.10.9`

`$ conda create -n wizardlm_langchain python=3.10.9`

Once the new environment is created, activate it.

`$ conda activate wizardlm_langchain`

### Setting up the environment

The entire process discussed above from 2 to 4 are automated using the [`setup.sh`](./setup.sh) bash script. Feel free to modify it according to your liking.

```bash
$ bash ./setup.sh
```

All the steps should ideally run without errors if the environment is setup correctly. 

If you are facing the following exception while running the demo:

```
Exception: Error while deserializing header: HeaderTooLarge
```

Then it means the model was not downloaded fully so you can try re-downloading it using the `git clone` command found in `setup.py`.

Before running the demo, it is good to deactivate and reactivate the environment when you are setting it up for the first time.

Run the demo:

`$ python demo.py`

Using the `setup.sh` will by default download the [wizardLM-7B-GPTQ](https://huggingface.co/TheBloke/wizardLM-7B-GPTQ) model but if you want to use other models that were tested with this project, you can use the `download_model.sh` script.

```bash
$ download_model.sh <HUGGING FACE MODEL NAME>
# Example
$ ./download_model.sh "TheBloke/WizardLM-7B-uncensored-GPTQ"
```

> **Note:** If you are unable to download the complete models from HF make sure Git LFS is correctly configured. The command `git lfs install` might sometimes get the job done.

## Usage

Once you have completed the setup process, you can use the GPTQ models with LangChain by following these steps:

> Make sure to append `wizardlm_langchain` project root dir to PYTHONPATH in order to use it globally

Refer to the example `demo.py` script to understand how to use it.

## Contributing

Contributions to this repository are welcome. If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

When contributing, please adhere to the following guidelines:

- Fork the repository and create a new branch for your contribution.
- Include documentation and comments where necessary.
- Write clear commit messages.
- Test your changes thoroughly before submitting a pull request.

## License

This repository is licensed under the GNU Public License. See the [LICENSE](./LICENSE) file for more information.

## Acknowledgements

We would like to acknowledge the contributions of the open-source community and the developers of the original GPTQ models used in this repository. A million thanks to [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui), their work has been of huge help for setting up GPTQ models with langchain.
