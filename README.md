# Fine-tuning with QLoRA

This repository fine-tunes `meta-llama/Llama-3.2-3B` with QLoRA using Hugging Face Transformers, TRL, PEFT, and Weights & Biases.

## Project structure

```text
.
├── README.md
└── src/
    └── fine-tuning/
        ├── create_Upload_dataset.py
        ├── fine-tuning-ollama.py
        ├── ft_model_testing.py
        └── util.py
```

## Files

- `src/fine-tuning/fine-tuning-ollama.py` — main training script for 4-bit QLoRA fine-tuning and Hub upload
- `src/fine-tuning/create_Upload_dataset.py` — dataset creation/upload helper
- `src/fine-tuning/ft_model_testing.py` — loads the fine-tuned model and runs evaluation
- `src/fine-tuning/util.py` — evaluation and plotting utilities

## Requirements

```bash
pip install -U transformers datasets accelerate bitsandbytes peft trl wandb huggingface_hub tqdm matplotlib scikit-learn pandas plotly
```

Use Google Colab with a CUDA GPU. You also need access to the base model on Hugging Face and a valid Hugging Face token.

## Setup

Update the constants in `src/fine-tuning/fine-tuning-ollama.py`:

```python
BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "your-wandb-project"
HF_USER = "your-huggingface-username"
DATA_USER = "dataset-owner"
DATASET_NAME = f"{DATA_USER}/dataset-name"
LITE_MODE = False
```

The dataset should contain:

- `train`
- `val`
- `test`

Store these in Colab Secrets:

- `HF_TOKEN`
- `WANDB_API_KEY`

## Run training

```bash
git clone https://github.com/khushboo13gupta/fine-tuning.git
cd fine-tuning
python src/fine-tuning/fine-tuning-ollama.py
```

The script:

- authenticates with Hugging Face and W&B
- loads the dataset
- loads the base model with 4-bit NF4 quantization
- applies LoRA adapters
- fine-tunes with `SFTTrainer`
- logs metrics to W&B
- uploads the trained model to the Hugging Face Hub

## Test the model

Adjust the run details in `src/fine-tuning/ft_model_testing.py`, then run:

```bash
python src/fine-tuning/ft_model_testing.py
```

This loads the fine-tuned adapter and evaluates outputs against the test split.
