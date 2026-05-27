"""Download the Bitext customer support dataset from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
FILENAME = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"


def main() -> int:
    """Download the CSV into the configured local data path."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/bitext_customer_support.csv")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
    output.write_bytes(Path(source).read_bytes())
    print(f"Downloaded {REPO_ID}/{FILENAME} to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
