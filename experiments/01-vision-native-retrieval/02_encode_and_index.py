"""02_encode_and_index.py -- encode page images with ColSmol on CPU and index in Qdrant.

Applies the verified MODEL-stream corrections:
  * transformers 5.3.0 (5.3.1 does not exist).
  * NO device_map='cpu' (that would require `accelerate`). Instead load then
    .to('cpu'). This keeps the dependency footprint minimal.
  * float32 + attn_implementation='eager' -> no flash-attn, pure CPU.

Applies the verified STORE-stream API:
  * multivector collection: VectorParams(size=DIM, distance=COSINE,
    multivector_config=MultiVectorConfig(comparator=MAX_SIM), hnsw_config m=0).
  * one PointStruct per page, vector={"colbert": [[...patch...], ...]}.

The per-patch dim is DERIVED from the first real embedding (colSmol projects to
128, but we never hardcode it).

Model fallback chain (set MODEL_ID env var to override):
  vidore/colSmol-256M  ->  vidore/colSmol-500M  (ColIdefics3 classes)
  vidore/colpali-v1.3  would use ColPali/ColPaliProcessor instead (see note).
"""
import os
import sys
import glob
import pickle

import torch
from PIL import Image

from colpali_engine.models import ColIdefics3, ColIdefics3Processor
from qdrant_client import QdrantClient, models

WORKDIR = os.path.dirname(os.path.abspath(__file__))
PAGES_DIR = os.path.join(WORKDIR, "pages")
EMB_CACHE = os.path.join(WORKDIR, "embeddings.pkl")  # for 03's fallback path

MODEL_ID = os.environ.get("MODEL_ID", "vidore/colSmol-256M")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = "vision_pages"
VECTOR_NAME = "colbert"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))  # 1 page at a time = CPU-RAM safe

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))


def collect_pages():
    """Return sorted list of (pmcid, page_index, png_path)."""
    items = []
    if not os.path.isdir(PAGES_DIR):
        raise SystemExit(f"ERROR: {PAGES_DIR} missing. Run 01_fetch_and_render.py first.")
    for pmcid in sorted(os.listdir(PAGES_DIR)):
        pdir = os.path.join(PAGES_DIR, pmcid)
        if not os.path.isdir(pdir):
            continue
        for png in sorted(glob.glob(os.path.join(pdir, "*.png"))):
            base = os.path.splitext(os.path.basename(png))[0]  # PMCxxxx_pNNN
            try:
                page_idx = int(base.rsplit("_p", 1)[1])
            except (IndexError, ValueError):
                page_idx = len(items)
            items.append((pmcid, page_idx, png))
    if not items:
        raise SystemExit(f"ERROR: no PNGs found under {PAGES_DIR}. Run step 01 first.")
    return items


def load_model():
    print(f"Loading {MODEL_ID} on CPU (float32, eager)... this can take a minute.")
    model = ColIdefics3.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to("cpu").eval()
    processor = ColIdefics3Processor.from_pretrained(MODEL_ID)
    return model, processor


@torch.no_grad()
def encode_images(model, processor, png_paths):
    """Return list of 2D python lists (n_patches x dim), one per image."""
    out = []
    for start in range(0, len(png_paths), BATCH_SIZE):
        chunk = png_paths[start:start + BATCH_SIZE]
        images = [Image.open(p).convert("RGB") for p in chunk]
        batch = processor.process_images(images).to(model.device)
        embs = model(**batch)  # (B, seq_len, dim), padded
        for row in embs:
            # row: (seq_len, dim) tensor -> drop pad rows is optional; MAX_SIM
            # with COSINE tolerates zero-pad patches, but we keep all rows.
            out.append(row.to(torch.float32).cpu().tolist())
        for im in images:
            im.close()
        print(f"  encoded {min(start + BATCH_SIZE, len(png_paths))}/{len(png_paths)} pages")
    return out


def ensure_collection(client, dim):
    exists = client.collection_exists(COLLECTION)
    if exists:
        print(f"Collection '{COLLECTION}' exists; recreating for a clean run.")
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            VECTOR_NAME: models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                hnsw_config=models.HnswConfigDiff(m=0),  # brute-force MAX_SIM rerank
            )
        },
    )
    print(f"Created multivector collection '{COLLECTION}' (dim={dim}, MAX_SIM).")


def main():
    pages = collect_pages()
    print(f"Found {len(pages)} page image(s) to encode.")

    model, processor = load_model()
    png_paths = [p for (_, _, p) in pages]
    vectors = encode_images(model, processor, png_paths)

    dim = len(vectors[0][0])
    print(f"Per-patch embedding dim = {dim}")

    # Cache embeddings + metadata for 03's no-Qdrant fallback.
    with open(EMB_CACHE, "wb") as f:
        pickle.dump(
            {"model_id": MODEL_ID, "dim": dim,
             "meta": [(pmcid, idx) for (pmcid, idx, _) in pages],
             "vectors": vectors},
            f,
        )
    print(f"Cached embeddings -> {EMB_CACHE} (used by 03 fallback).")

    # --- Qdrant indexing (optional; fail soft with a clear message) ---
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=60)
        client.get_collections()  # connectivity probe
    except Exception as e:  # noqa: BLE001
        print(
            f"\nWARNING: Qdrant unreachable at {QDRANT_URL} ({e}).\n"
            f"Embeddings are cached; run 03_query.py --fallback to sanity-check "
            f"retrieval without Qdrant. Start Qdrant then re-run this script to index.",
            file=sys.stderr,
        )
        sys.exit(2)

    ensure_collection(client, dim)
    points = [
        models.PointStruct(
            id=i,
            vector={VECTOR_NAME: vectors[i]},
            payload={"pmcid": pages[i][0], "page": pages[i][1]},
        )
        for i in range(len(pages))
    ]
    # upsert in small batches to keep gRPC/HTTP payloads modest
    for start in range(0, len(points), 8):
        client.upsert(collection_name=COLLECTION, points=points[start:start + 8])
    count = client.count(COLLECTION).count
    print(f"\nDONE: upserted {count} page point(s) into Qdrant collection '{COLLECTION}'.")


if __name__ == "__main__":
    main()
