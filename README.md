# Fine-tuning with QLoRA

This repository contains experiments for fine-tuning large language models with **QLoRA**, using Hugging Face Transformers, TRL, PEFT, Weights & Biases, and Google Colab.

The current training script fine-tunes `meta-llama/Llama-3.2-3B` using 4-bit quantization and LoRA adapters, then uploads the resulting private model to the Hugging Face Hub.

## Features

- 4-bit NF4 quantization with BitsAndBytes
- LoRA adapters applied to the attention projection layers
- Supervised fine-tuning with TRL's `SFTTrainer`
- Dataset loading from the Hugging Face Hub
- Experiment tracking with Weights & Biases
- Automatic model uploads to the Hugging Face Hub
- Automatic selection of BF16 or FP16 based on GPU capability

## Project structure

```text
.
├── src/
│   └── fine-tuning/
│       └── fine-tuning-ollama.py
└── README.md
```

## Requirements

The script is designed to run in Google Colab and requires a CUDA-enabled GPU. Install the required packages before running it:

```bash
pip install -U transformers datasets accelerate bitsandbytes peft trl wandb huggingface_hub tqdm matplotlib
```

You also need access to the selected gated model on Hugging Face and a compatible GPU with sufficient VRAM.

## Configuration

Before running the script, update the project constants in `src/fine-tuning/fine-tuning-ollama.py`:

```python
BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "your-wandb-project"
HF_USER = "your-huggingface-username"
DATA_USER = "dataset-owner"
DATASET_NAME = f"{DATA_USER}/dataset-name"
```

The dataset is expected to contain these splits:

- `train`
- `val`
- `test`

The training and validation records should be compatible with the format expected by TRL's `SFTTrainer`. For example, a conversational dataset may contain a `messages` field:

```json
{"messages":[{"role":"user","content":"What is machine learning?"},{"role":"assistant","content":"Machine learning is a method for learning patterns from data."}]}
```

## Authentication

Store the following secrets in Google Colab under **Secrets**:

- `HF_TOKEN` — a Hugging Face access token with permission to read the base model and write the fine-tuned model
- `WANDB_API_KEY` — a Weights & Biases API key

The script reads these values with `google.colab.userdata`. Do not hard-code tokens or commit them to the repository.

You must also accept the model license and request access to `meta-llama/Llama-3.2-3B` on Hugging Face before downloading it.

## Running the training script

In Google Colab, clone the repository and run:

```bash
git clone https://github.com/khushboo13gupta/fine-tuning.git
cd fine-tuning
python src/fine-tuning/fine-tuning-ollama.py
```

The script will:

1. Authenticate with Hugging Face and Weights & Biases.
2. Load the dataset from the Hugging Face Hub.
3. Load the base model using 4-bit quantization.
4. Configure LoRA adapters for the attention layers.
5. Fine-tune the model with `SFTTrainer`.
6. Log training metrics to Weights & Biases.
7. Push checkpoints and the final private model to the Hugging Face Hub.

## Main hyperparameters

The default configuration uses:

| Parameter | Default |
| --- | ---: |
| Epochs | `1` |
| Per-device batch size | `32` |
| Maximum sequence length | `128` |
| Learning rate | `1e-4` |
| LoRA rank | `32` |
| LoRA alpha | `64` |
| LoRA dropout | `0.1` |
| Quantization | 4-bit NF4 |
| Scheduler | cosine |

Adjust the batch size, sequence length, gradient accumulation, and quantization settings according to the available GPU memory.

## Important notes

- This script currently references `LITE_MODE` when configuring validation and logging, but `LITE_MODE` must be defined before execution. Add, for example, `LITE_MODE = False` near the other constants.
- A batch size of `32` may exceed the memory available on many GPUs. Reduce `BATCH_SIZE` or increase `GRADIENT_ACCUMULATION_STEPS` if you encounter out-of-memory errors.
- The `test` split is loaded but is not yet used for final evaluation in the script.
- Keep model checkpoints and private datasets out of version control.
- Training costs depend on the selected GPU, dataset size, and number of epochs.

## License

No license has been specified for this repository yet. Add a license before distributing the code, datasets, or trained model artifacts.