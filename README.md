# BeLink-Reranker

A generative re-ranker instruction-tuned for biomedical entity linking / candidate reranking.
Given a sentence, an entity mention, and a list of candidate concepts, the model
selects the correct concept (or "none of the above").

## Installation

This package depends on a **modified fork of `ms-swift`** and a **specific CUDA
build of PyTorch**. Install torch first, matching your CUDA setup, then install
this package.

```bash
# 1. Install torch matching CUDA 12.6 (adjust if your system differs)
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# 2. Install swift-biolink (pulls in the modified swift fork + remaining deps)
pip install git+https://github.com/dash-ka/swift-biolink.git
```

If your machine uses a different CUDA version, find the matching torch install
command at https://pytorch.org/get-started/locally/, run that first, then
proceed to step 2.

## Model weights

Model weights are hosted on Hugging Face at
[`Dash00/belink-reranker`](https://huggingface.co/Dash00/belink-reranker) and are
downloaded automatically on first use (cached locally under
`~/.cache/belink_reranker/` by default, or the directory set in the
`BELINK_RERANKER_CACHE` environment variable).

## Usage

```python
from belink_qwen import BelinkReranker, rerank

model = BelinkReranker.load()  # downloads weights on first run

items = [
    (
        "The patient was diagnosed with type 2 diabetes mellitus.",
        "type 2 diabetes mellitus",
        ["D003924>Diabetes Mellitus, Type 2", "D003920>Diabetes Mellitus"],
    ),
]

predictions = rerank(model, items)
print(predictions)
```

### Reranking a BioC XML file directly

```python
from belink_qwen import BelinkReranker, rerank_biocxml_file

model = BelinkReranker.load()
rerank_biocxml_file(model, "path/to/file.bioc.xml")
```

### Choosing a GPU

```python
model = BelinkReranker.load()
```

## License

Apache 2.0. This project depends on a modified fork of
[ms-swift](https://github.com/dash-ka/swift-biolink), also Apache 2.0.
