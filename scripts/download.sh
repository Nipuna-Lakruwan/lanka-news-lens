#!/usr/bin/env bash
# Download the pinned revisions of nuuuwan's datasets into data/raw/ (about 250 MB).
set -euo pipefail
mkdir -p data/raw && cd data/raw
B=https://huggingface.co/datasets/nuuuwan
DOCS=efd995a871303c05a6df27fdf2a799e99a77b51e
CHUNKS=390ba0ff687771247b6216fc2f84e1699e209e9b
curl -fL -o docs.parquet "$B/lk-news-docs/resolve/$DOCS/data/train-00000-of-00001.parquet"
for i in 0 1; do
  curl -fL -o chunks-$i.parquet "$B/lk-news-chunks/resolve/$CHUNKS/data/train-0000$i-of-00002.parquet"
done
printf 'lk-news-docs %s\nlk-news-chunks %s\n' "$DOCS" "$CHUNKS" > REVISIONS.txt
