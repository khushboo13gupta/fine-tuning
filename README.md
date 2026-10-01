# Fine-tuning with QLoRA

This repository demonstrates how to fine-tune Meta's Llama 3.2 3B model with QLoRA using Hugging Face Transformers, TRL, PEFT, and Weights & Biases. It includes everything needed to prepare a dataset, train a low-rank adapted model, evaluate it on a held-out test split, and optionally deploy the model through Modal for inference.

## Project structure

```text
.
├── README.md
└── src/
    └── fine-tuning/
        ├── create_Upload_dataset.py
        ├── fine-tuning-ollama.py
        ├── ft_model_testing.py
        ├── modal_deployment_testing.py
        └── util.py
```

## Files

- `src/fine-tuning/fine-tuning-ollama.py` — main training script for 4-bit QLoRA fine-tuning and Hub upload
- `src/fine-tuning/create_Upload_dataset.py` — dataset creation/upload helper
- `src/fine-tuning/ft_model_testing.py` — loads the fine-tuned model and runs evaluation
- `src/fine-tuning/modal_deployment_testing.py` — Modal deployment stub with pricing method
- `src/fine-tuning/util.py` — evaluation and plotting utilities

## Requirements

```bash
pip install -U transformers datasets accelerate bitsandbytes peft trl wandb huggingface_hub tqdm matplotlib scikit-learn pandas plotly modal
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

## Modal Deployment Testing

### Setup Modal

1. Install Modal CLI:
```bash
pip install modal
```

2. Authenticate with Modal:
```bash
modal token new
```

3. Add your Hugging Face secret to Modal:
```bash
modal secret create huggingface-secret --key HF_TOKEN --value <your-hf-token>
```

### Update Modal Deployment Configuration

Edit `src/fine-tuning/modal_deployment_testing.py` and update:

```python
PROJECT_NAME = "your-project-name"
FINETUNED_MODEL = "your-hf-username/your-finetuned-model-name"
```

### Deploy to Modal

Deploy your Modal app:

```bash
modal deploy src/fine-tuning/modal_deployment_testing.py
```

### Test Modal Deployment

Test the deployment with a sample price prediction:

```bash
modal run src/fine-tuning/modal_deployment_testing.py::Pricer.price --input "Product: Laptop, Brand: Dell, Specs: 16GB RAM, 512GB SSD"
```

Or test interactively:

```python
import modal

app = modal.App.lookup("pricer-service")
pricer = app["Pricer"]()

price = pricer.price.remote("Product: Laptop, Brand: Dell, Specs: 16GB RAM, 512GB SSD")
print(f"Predicted price: ${price}")
```

### Call Price Method

To call the `price()` method on your deployed Modal service:

**From command line:**

```bash
modal run src/fine-tuning/modal_deployment_testing.py::Pricer.price --input "Your product description here"
```

**From Python script:**

```python
import modal

# Lookup the deployed app
app = modal.App.lookup("pricer-service")
pricer = app["Pricer"]()

# Call the price method
description = "Product: Used Bicycle, Brand: Trek, Condition: Good"
price = pricer.price.remote(description)
print(f"Predicted price: ${price:.2f}")
```

**Get Modal pricing information:**

```bash
modal pricing
```

**List running Modal deployments:**

```bash
modal list apps
```

**View deployment logs:**

```bash
modal logs <deployment-name>
```

**Monitor deployment metrics:**

```bash
modal get-metrics <deployment-name>
```

### Example Usage

Here's a complete example to test the pricing model:

```python
import modal

app = modal.App.lookup("pricer-service")
pricer_class = app["Pricer"]()

# Test with multiple items
test_items = [
    "iPhone 15 Pro Max, 256GB, Space Black, excellent condition",
    "Vintage wooden desk, solid oak, 48x24 inches",
    "Mountain bike, hybrid, aluminum frame, 21 speeds"
]

for item in test_items:
    price = pricer_class.price.remote(item)
    print(f"Item: {item}")
    print(f"Predicted Price: ${price:.2f}\n")
```
