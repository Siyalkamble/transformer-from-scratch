"""Downloads TinyStories dataset into data/ (gitignored)."""
from datasets import load_dataset
import json, os

os.makedirs("data", exist_ok=True)
ds = load_dataset("roneneldan/TinyStories")
ds["train"].to_json("data/tinystories_train.jsonl")
ds["validation"].to_json("data/tinystories_val.jsonl")