"""Force HuggingFace tests to stay offline so CI never downloads DistilBERT.

Live `distilbert-base-uncased` training is `scripts/train_distilbert.py`,
not pytest. These environment variables are set before test modules import
transformers.
"""

# Import os to export Hub-offline flags before transformers is imported.
import os

# Refuse Hub downloads for the entire pytest session (tiny local models only).
os.environ["HF_HUB_OFFLINE"] = "1"

# Refuse transformers weight downloads even if a test calls from_pretrained.
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Disable wandb if an operator has it installed globally.
os.environ["WANDB_DISABLED"] = "true"

# Keep tokenizers single-threaded so WSL2 does not emit fork warnings.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
