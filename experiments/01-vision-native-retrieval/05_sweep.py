"""05_sweep.py -- model-parameterized retrieval sweep (CPU-only).

For MODEL_ID (env), encode all 51 pages (cache per-model so nothing clobbers),
run the fixed paper-scoped queries in queries.json via score_multi_vector
(already proven == Qdrant MAX_SIM), and write results_<model>.json with top-10.

Usage:
  set MODEL_ID=vidore/colSmol-256M   # reuses embeddings_colSmol-256M.pkl (fast)
  set MODEL_ID=vidore/colSmol-500M   # encodes fresh (slow, ~minutes on CPU)
  python 05_sweep.py
"""
import os
import sys
import glob
import json
import pickle

import torch
from PIL import Image

WORKDIR = os.path.dirname(os.path.abspath(__file__))
PAGES_DIR = os.path.join(WORKDIR, "pages")
MODEL_ID = os.environ.get("MODEL_ID", "vidore/colSmol-500M")
SAFE = MODEL_ID.split("/")[-1]
EMB_CACHE = os.path.join(WORKDIR, f"embeddings_{SAFE}.pkl")
RESULTS = os.path.join(WORKDIR, f"results_{SAFE}.json")
TOPK = 10
BATCH = int(os.environ.get("BATCH_SIZE", "1"))

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
torch.set_grad_enabled(False)


def load_model():
    ml = MODEL_ID.lower()
    if "colsmol" in ml or "idefics" in ml:
        from colpali_engine.models import ColIdefics3 as M, ColIdefics3Processor as P
    elif "colqwen2.5" in ml or "colqwen2_5" in ml or "colnomic" in ml:
        # colnomic-embed-multimodal-3b/7b are Qwen2.5-VL-based; load via ColQwen2_5 classes
        from colpali_engine.models import ColQwen2_5 as M, ColQwen2_5_Processor as P
    elif "colpali" in ml:
        from colpali_engine.models import ColPali as M, ColPaliProcessor as P
    else:
        raise SystemExit(f"unknown model family for {MODEL_ID}")
    print(f"Loading {MODEL_ID} on CPU (float32, eager)...")
    model = M.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32,
        attn_implementation="eager", low_cpu_mem_usage=True,
    ).to("cpu").eval()
    return model, P.from_pretrained(MODEL_ID)


def collect_pages():
    items = []
    for pmcid in sorted(os.listdir(PAGES_DIR)):
        pdir = os.path.join(PAGES_DIR, pmcid)
        if not os.path.isdir(pdir):
            continue
        for png in sorted(glob.glob(os.path.join(pdir, "*.png"))):
            idx = int(os.path.splitext(os.path.basename(png))[0].rsplit("_p", 1)[1])
            items.append((pmcid, idx, png))
    return items


def main():
    pages = collect_pages()
    meta = [(p, i) for (p, i, _) in pages]
    model, proc = load_model()

    if os.path.exists(EMB_CACHE):
        print(f"Using cached page embeddings: {EMB_CACHE}")
        cache = pickle.load(open(EMB_CACHE, "rb"))
        page_embs = [torch.tensor(v, dtype=torch.float32) for v in cache["vectors"]]
        meta = [tuple(m) for m in cache["meta"]]
    else:
        print(f"Encoding {len(pages)} pages with {MODEL_ID} (CPU, batch={BATCH})...")
        paths = [p for (_, _, p) in pages]
        page_embs = []
        for s in range(0, len(paths), BATCH):
            imgs = [Image.open(x).convert("RGB") for x in paths[s:s + BATCH]]
            batch = proc.process_images(imgs).to("cpu")
            emb = model(**batch)
            for row in emb:
                page_embs.append(row.to(torch.float32).cpu())
            for im in imgs:
                im.close()
            del batch, emb
            print(f"  encoded {min(s + BATCH, len(paths))}/{len(paths)}")
        pickle.dump(
            {"model_id": MODEL_ID, "dim": int(page_embs[0].shape[-1]),
             "meta": meta, "vectors": [e.tolist() for e in page_embs]},
            open(EMB_CACHE, "wb"),
        )
        print(f"Cached -> {EMB_CACHE}")

    queries = json.load(open(os.path.join(WORKDIR, "queries.json"), encoding="utf-8"))
    out = {"model": MODEL_ID, "dim": int(page_embs[0].shape[-1]), "queries": {}}
    for q in queries:
        qb = proc.process_queries([q["text"]]).to("cpu")
        qe = model(**qb)
        scores = proc.score_multi_vector(qe, page_embs)[0]
        tk = torch.topk(scores, k=min(TOPK, len(meta)))
        ranked = [{"pmcid": meta[i][0], "page": meta[i][1], "score": round(float(v), 4)}
                  for v, i in zip(tk.values.tolist(), tk.indices.tolist())]
        out["queries"][q["id"]] = ranked
        print(f"  {q['id']} [{q['paper']}] top3: "
              + ", ".join(f"{r['pmcid']}:p{r['page']}({r['score']})" for r in ranked[:3]))
    json.dump(out, open(RESULTS, "w", encoding="utf-8"), indent=2)
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
