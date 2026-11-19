#!/usr/bin/env bash
# Upload the fine-tuned topic classifier to the Hugging Face Hub.
# Run `hf auth login` yourself first (this script never handles tokens).
set -euo pipefail
cd "$(dirname "$0")"
REPO="nipuna02/lk-news-topic-classifier"
MODEL_DIR="models/lk-news-topic-classifier"
[ -f "$MODEL_DIR/model.safetensors" ] || { echo "No model at $MODEL_DIR. Run scripts/finetune.py first."; exit 1; }
cp model_card/README.md "$MODEL_DIR/README.md"
cp results/topic_finetuned.json "$MODEL_DIR/eval_results.json"
hf auth whoami >/dev/null || { echo "Not logged in. Run: hf auth login"; exit 1; }
hf repos create "$REPO" --repo-type model --exist-ok
hf upload "$REPO" "$MODEL_DIR" . --commit-message "Upload lk-news-topic-classifier"
echo "Done: https://huggingface.co/$REPO"
