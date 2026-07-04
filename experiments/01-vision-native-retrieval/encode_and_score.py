"""
Stream D PoC: colpali-engine encode + MaxSim score sanity check (CPU-only).

Target: Marathon (Ryzen AI MAX+ 395, no NVIDIA/CUDA). Runs torch CPU-only.
Model: vidore/colSmol-256M  (ColIdefics3 / SmolVLM-256M backbone, 128-dim multivector)
Purpose: verify late-interaction retrieval independently of Qdrant.

Env (run in a uv venv pinned to Python 3.12):
    uv venv --python 3.12
    uv pip install "colpali-engine==0.3.17" "torch>=2.2.0,<2.12.0" pillow
"""

import glob
import torch
from PIL import Image
from colpali_engine.models import ColIdefics3, ColIdefics3Processor

MODEL_NAME = "vidore/colSmol-256M"
BATCH_SIZE = 2                 # keep small; CPU + 128GB but avoid RAM spikes on big page images
IMAGE_GLOB = "pages/*.png"     # a handful of rendered page images
QUERY = "What was the primary outcome and effect size reported?"

# --- Force CPU, single-thread-friendly, deterministic float32 -----------------
torch.set_grad_enabled(False)   # global; we also wrap encode in no_grad
DEVICE = "cpu"


def load_model():
    # NO device_map (that path assumes accelerate/GPU placement); .to("cpu") is explicit.
    # NO attn_implementation="flash_attention_2" (CUDA-only kernel).
    # float32 on CPU: bfloat16 CPU matmul is slow/unsupported for parts of the stack.
    model = ColIdefics3.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=True,
    ).to(DEVICE).eval()
    processor = ColIdefics3Processor.from_pretrained(MODEL_NAME)
    return model, processor


def encode_images(model, processor, images):
    """Returns a list of (num_patches, 128) float32 tensors, one per page image."""
    doc_embs = []
    for i in range(0, len(images), BATCH_SIZE):
        chunk = images[i : i + BATCH_SIZE]
        batch = processor.process_images(chunk).to(DEVICE)
        with torch.no_grad():
            emb = model(**batch)              # (B, num_patches, 128), padded per batch
        # split batch into per-image tensors so scoring gets a clean list
        doc_embs.extend(list(torch.unbind(emb, dim=0)))
        del batch, emb                        # release RAM between chunks
    return doc_embs


def encode_queries(model, processor, queries):
    """Returns a list of (num_query_tokens, 128) float32 tensors."""
    batch = processor.process_queries(queries).to(DEVICE)
    with torch.no_grad():
        emb = model(**batch)                  # (Q, num_query_tokens, 128)
    return list(torch.unbind(emb, dim=0))


def main():
    paths = sorted(glob.glob(IMAGE_GLOB))
    assert paths, f"No images matched {IMAGE_GLOB}"
    images = [Image.open(p).convert("RGB") for p in paths]

    model, processor = load_model()

    doc_embs = encode_images(model, processor, images)
    query_embs = encode_queries(model, processor, [QUERY])

    print("doc multivector shape (first page):", tuple(doc_embs[0].shape))   # (num_patches, 128)
    print("query multivector shape:", tuple(query_embs[0].shape))            # (num_tokens, 128)

    # MaxSim / ColBERT late-interaction scoring. Accepts lists of variable-length
    # multivectors; handles padding internally. Returns (num_queries, num_docs).
    scores = processor.score_multi_vector(query_embs, doc_embs)
    print("scores shape:", tuple(scores.shape))                              # (1, num_pages)

    ranking = scores[0].argsort(descending=True)
    print("\nRanked pages for query:", QUERY)
    for rank, idx in enumerate(ranking.tolist(), 1):
        print(f"  {rank}. {paths[idx]}   score={scores[0, idx].item():.4f}")


if __name__ == "__main__":
    main()
