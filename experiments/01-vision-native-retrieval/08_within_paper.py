"""08_within_paper.py -- the reviewer's cheap decisive probe.

paper@1 = 5/5, so document routing is solved. The open question is whether
ColPali is usable as a *within-paper page reranker*: if we restrict candidates
to the correct paper, does the gold results-table page rank near the top?

Reuses cached page embeddings (no re-encoding); only re-encodes the 5 queries.
For each model x query: rank ONLY the gold paper's pages, report the 1-based
position of the first gold page among that paper's N pages.
"""
import os
import glob
import json
import pickle

import torch

WORKDIR = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
torch.set_grad_enabled(False)


def load_model(model_id):
    ml = model_id.lower()
    if "colsmol" in ml or "idefics" in ml:
        from colpali_engine.models import ColIdefics3 as M, ColIdefics3Processor as P
    elif "colqwen2.5" in ml or "colqwen2_5" in ml:
        from colpali_engine.models import ColQwen2_5 as M, ColQwen2_5_Processor as P
    else:
        from colpali_engine.models import ColPali as M, ColPaliProcessor as P
    model = M.from_pretrained(model_id, torch_dtype=torch.float32,
                              attn_implementation="eager", low_cpu_mem_usage=True).to("cpu").eval()
    return model, P.from_pretrained(model_id)


def main():
    queries = json.load(open(os.path.join(WORKDIR, "queries.json"), encoding="utf-8"))
    gold = {l["query_id"]: l for l in json.load(open(os.path.join(WORKDIR, "gold_labels.json"), encoding="utf-8"))["labels"]}

    caches = sorted(glob.glob(os.path.join(WORKDIR, "embeddings_*.pkl")))
    summary = {}
    for cpath in caches:
        cache = pickle.load(open(cpath, "rb"))
        model_id = cache["model_id"]
        meta = [tuple(m) for m in cache["meta"]]
        page_embs = [torch.tensor(v, dtype=torch.float32) for v in cache["vectors"]]
        print(f"\n=== {model_id} (within-paper reranking) ===")
        model, proc = load_model(model_id)

        hits1 = hits3 = n = 0
        per_q = {}
        for q in queries:
            g = gold[q["id"]]
            gold_pages = set(g["gold_pages"])
            paper = g["pmcid"]
            qb = proc.process_queries([q["text"]]).to("cpu")
            qe = model(**qb)
            scores = proc.score_multi_vector(qe, page_embs)[0]
            # restrict to this paper's pages
            idxs = [i for i, m in enumerate(meta) if m[0] == paper]
            ranked = sorted(idxs, key=lambda i: float(scores[i]), reverse=True)
            npages = len(idxs)
            rank = next((r for r, i in enumerate(ranked, 1) if meta[i][1] in gold_pages), None)
            per_q[q["id"]] = {"paper": paper, "within_rank": rank, "of": npages}
            n += 1
            hits1 += int(rank == 1)
            hits3 += int(rank is not None and rank <= 3)
            print(f"  {q['id']} [{paper}] gold page rank {rank} of {npages} pages")
        summary[model_id] = {"within_hit@1": f"{hits1}/{n}", "within_hit@3": f"{hits3}/{n}", "per_q": per_q}
        del model, proc

    print("\n\n=== WITHIN-PAPER SUMMARY (candidates restricted to correct paper) ===")
    for mid, s in summary.items():
        print(f"  {mid:28} within-hit@1 {s['within_hit@1']}   within-hit@3 {s['within_hit@3']}")
    json.dump(summary, open(os.path.join(WORKDIR, "within_paper_report.json"), "w", encoding="utf-8"), indent=2)
    print("\nwrote within_paper_report.json")


if __name__ == "__main__":
    main()
