"""03_query.py -- encode a text query and retrieve top-k (pmcid, page) pages.

Usage:
  python 03_query.py "What was the primary endpoint?"            # Qdrant path
  python 03_query.py --fallback "What was the primary endpoint?" # no-Qdrant path
  python 03_query.py -k 10 "..."                                  # top-k override

Applies verified corrections:
  * MODEL: load with .to('cpu') (no device_map -> no accelerate needed).
  * STORE: client.query_points(collection_name=..., query=<2D list>,
    using='colbert', limit=k, with_payload=True); read resp.points[i].score.
  * FALLBACK: processor.score_multi_vector(query_embs, page_embs) -- the
    library's own MAX_SIM, run against cached page embeddings from step 02.
"""
import os
import sys
import pickle
import argparse

import torch

from colpali_engine.models import ColIdefics3, ColIdefics3Processor

WORKDIR = os.path.dirname(os.path.abspath(__file__))
EMB_CACHE = os.path.join(WORKDIR, "embeddings.pkl")

MODEL_ID = os.environ.get("MODEL_ID", "vidore/colSmol-256M")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = "vision_pages"
VECTOR_NAME = "colbert"

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))


def load_model():
    print(f"Loading {MODEL_ID} on CPU (float32, eager)...")
    model = ColIdefics3.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to("cpu").eval()
    processor = ColIdefics3Processor.from_pretrained(MODEL_ID)
    return model, processor


@torch.no_grad()
def encode_query(model, processor, text):
    batch = processor.process_queries([text]).to(model.device)
    embs = model(**batch)  # (1, q_len, dim)
    return embs  # keep as tensor for fallback; convert to list for Qdrant


def run_qdrant(query_embs, k):
    from qdrant_client import QdrantClient
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=60)
        client.get_collections()
    except Exception as e:  # noqa: BLE001
        print(
            f"ERROR: Qdrant unreachable at {QDRANT_URL} ({e}).\n"
            f"Start it (docker run ... qdrant/qdrant) or use --fallback.",
            file=sys.stderr,
        )
        sys.exit(2)
    if not client.collection_exists(COLLECTION):
        print(
            f"ERROR: collection '{COLLECTION}' does not exist. Run 02_encode_and_index.py.",
            file=sys.stderr,
        )
        sys.exit(2)
    query_2d = query_embs[0].to(torch.float32).cpu().tolist()  # (q_len, dim)
    resp = client.query_points(
        collection_name=COLLECTION,
        query=query_2d,
        using=VECTOR_NAME,
        limit=k,
        with_payload=True,
    )
    print(f"\nTop-{k} (Qdrant MAX_SIM):")
    for rank, p in enumerate(resp.points, 1):
        pl = p.payload or {}
        print(f"  {rank:2d}. {pl.get('pmcid')}  page {pl.get('page')}  score={p.score:.4f}")


def run_fallback(model, processor, query_embs, k):
    if not os.path.exists(EMB_CACHE):
        print(
            f"ERROR: {EMB_CACHE} not found. Run 02_encode_and_index.py first "
            f"(it caches embeddings even if Qdrant is down).",
            file=sys.stderr,
        )
        sys.exit(2)
    with open(EMB_CACHE, "rb") as f:
        cache = pickle.load(f)
    meta = cache["meta"]
    page_embs = [torch.tensor(v, dtype=torch.float32) for v in cache["vectors"]]
    # processor.score_multi_vector accepts lists of 2D tensors of differing len.
    scores = processor.score_multi_vector(query_embs, page_embs)  # (1, n_pages)
    scores = scores[0]
    topk = torch.topk(scores, k=min(k, len(meta)))
    print(f"\nTop-{k} (library score_multi_vector fallback, no Qdrant):")
    for rank, (val, idx) in enumerate(zip(topk.values.tolist(), topk.indices.tolist()), 1):
        pmcid, page = meta[idx]
        print(f"  {rank:2d}. {pmcid}  page {page}  score={val:.4f}")


def main():
    ap = argparse.ArgumentParser(description="Retrieve top-k PMC pages for a text query.")
    ap.add_argument("query", nargs="?", default="What was the primary endpoint of the study?")
    ap.add_argument("-k", type=int, default=5, help="top-k results (default 5)")
    ap.add_argument("--fallback", action="store_true",
                    help="skip Qdrant; score against cached embeddings via score_multi_vector")
    args = ap.parse_args()

    print(f"Query: {args.query!r}")
    model, processor = load_model()
    query_embs = encode_query(model, processor, args.query)

    if args.fallback:
        run_fallback(model, processor, query_embs, args.k)
    else:
        run_qdrant(query_embs, args.k)


if __name__ == "__main__":
    main()
