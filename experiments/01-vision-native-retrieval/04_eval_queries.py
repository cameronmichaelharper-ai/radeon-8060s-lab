"""04_eval_queries.py -- batch retrieval eval: load model once, run several
clinical table-fact queries through BOTH Qdrant (MAX_SIM) and the library's own
score_multi_vector fallback, and report top-k agreement.

The queries target numeric/results-table facts -- the exact class of question the
Citadel text-chunking eval got wrong before table-aware chunking (2/9 -> 8/9).
Here we check whether vision-native page retrieval lands on the right *page*
without any text extraction or chunking at all.
"""
import os
import pickle

import torch
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
from qdrant_client import QdrantClient

WORKDIR = os.path.dirname(os.path.abspath(__file__))
EMB_CACHE = os.path.join(WORKDIR, "embeddings.pkl")
MODEL_ID = os.environ.get("MODEL_ID", "vidore/colSmol-256M")
COLLECTION = "vision_pages"
VECTOR_NAME = "colbert"
K = 5

# (label, query) -- each answer lives in a specific paper's results/table.
QUERIES = [
    ("FMD glycemic effect",
     "What was the effect of the fasting-mimicking diet on HbA1c or fasting glucose?"),
    ("Digital HTN BP reduction",
     "Pooled reduction in systolic blood pressure from digital health interventions for hypertension"),
    ("Personalized nutrition primary outcome",
     "Primary outcome result of the personalized nutrition randomized trial, postprandial glucose"),
    ("Sample size / randomization",
     "How many participants were randomized and what were the group allocations?"),
    ("Adverse events",
     "What adverse events or safety outcomes were reported?"),
]

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))


def load_model():
    print(f"Loading {MODEL_ID} on CPU...")
    model = ColIdefics3.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, attn_implementation="eager"
    ).to("cpu").eval()
    processor = ColIdefics3Processor.from_pretrained(MODEL_ID)
    return model, processor


@torch.no_grad()
def encode_query(model, processor, text):
    batch = processor.process_queries([text]).to(model.device)
    return model(**batch)  # (1, q_len, dim)


def main():
    with open(EMB_CACHE, "rb") as f:
        cache = pickle.load(f)
    meta = cache["meta"]
    page_embs = [torch.tensor(v, dtype=torch.float32) for v in cache["vectors"]]

    client = QdrantClient(url="http://localhost:6333", timeout=60)
    model, processor = load_model()

    agree = 0
    for label, q in QUERIES:
        qe = encode_query(model, processor, q)

        # Qdrant MAX_SIM
        q2d = qe[0].to(torch.float32).cpu().tolist()
        resp = client.query_points(
            collection_name=COLLECTION, query=q2d, using=VECTOR_NAME,
            limit=K, with_payload=True,
        )
        qd = [(p.payload["pmcid"], p.payload["page"], p.score) for p in resp.points]

        # Library fallback
        scores = processor.score_multi_vector(qe, page_embs)[0]
        topk = torch.topk(scores, k=K)
        fb = [(*meta[i], float(v)) for v, i in zip(topk.values.tolist(), topk.indices.tolist())]

        top_match = (qd[0][0], qd[0][1]) == (fb[0][0], fb[0][1])
        agree += int(top_match)

        print(f"\n=== {label} ===\n  Q: {q}")
        print(f"  Qdrant top-{K}:  " + "  ".join(f"{m}:p{pg}({s:.2f})" for m, pg, s in qd))
        print(f"  Library top-{K}: " + "  ".join(f"{m}:p{pg}({s:.2f})" for m, pg, s in fb))
        print(f"  top-1 agree: {top_match}")

    print(f"\nQdrant vs library top-1 agreement: {agree}/{len(QUERIES)}")


if __name__ == "__main__":
    main()
