"""06_score.py -- score each model's retrieval against gold labels.

Inputs:
  gold_labels.json  -> {"labels": [{query_id, pmcid, gold_pages:[int], answer_value, confidence}, ...]}
  results_<model>.json (one per model) -> {"model", "queries": {qid: [{pmcid, page, score}, ...]}}

Metrics per model (over the 5 queries):
  hit@1 / hit@3 / hit@5  : a gold (pmcid, page) appears in top-k
  MRR                    : mean reciprocal rank of first gold hit
  paper@1                : top-1 is at least the correct paper (page may be wrong)

Writes score_report.json and prints a comparison table.
"""
import os
import glob
import json

WORKDIR = os.path.dirname(os.path.abspath(__file__))


def load_gold():
    g = json.load(open(os.path.join(WORKDIR, "gold_labels.json"), encoding="utf-8"))
    out = {}
    for lab in g["labels"]:
        out[lab["query_id"]] = {"pmcid": lab["pmcid"], "pages": set(lab.get("gold_pages", []))}
    return out


def first_hit_rank(ranked, gold):
    for i, r in enumerate(ranked, 1):
        if r["pmcid"] == gold["pmcid"] and r["page"] in gold["pages"]:
            return i
    return None


def score_model(res, gold):
    per_q = {}
    agg = {"hit@1": 0, "hit@3": 0, "hit@5": 0, "mrr": 0.0, "paper@1": 0, "n": 0}
    for qid, ranked in res["queries"].items():
        if qid not in gold:
            continue
        g = gold[qid]
        rank = first_hit_rank(ranked, g)
        paper1 = bool(ranked) and ranked[0]["pmcid"] == g["pmcid"]
        rr = (1.0 / rank) if rank else 0.0
        per_q[qid] = {
            "gold_pages": sorted(g["pages"]), "gold_pmcid": g["pmcid"],
            "first_hit_rank": rank, "paper@1": paper1,
            "top1": (ranked[0] if ranked else None),
        }
        agg["n"] += 1
        agg["hit@1"] += int(rank == 1)
        agg["hit@3"] += int(rank is not None and rank <= 3)
        agg["hit@5"] += int(rank is not None and rank <= 5)
        agg["mrr"] += rr
        agg["paper@1"] += int(paper1)
    if agg["n"]:
        agg["mrr"] = round(agg["mrr"] / agg["n"], 4)
    return per_q, agg


def main():
    gold = load_gold()
    report = {}
    for rf in sorted(glob.glob(os.path.join(WORKDIR, "results_*.json"))):
        res = json.load(open(rf, encoding="utf-8"))
        model = res["model"]
        per_q, agg = score_model(res, gold)
        report[model] = {"agg": agg, "per_query": per_q}

    json.dump(report, open(os.path.join(WORKDIR, "score_report.json"), "w", encoding="utf-8"), indent=2)

    n = len(gold)
    print(f"\nGold labels: {n} queries")
    for qid in sorted(gold):
        print(f"  {qid}: {gold[qid]['pmcid']} pages {sorted(gold[qid]['pages'])}")
    hdr = f"\n{'model':28} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'MRR':>6} {'paper@1':>8}"
    print(hdr); print("-" * len(hdr))
    for model, r in report.items():
        a = r["agg"]
        print(f"{model:28} {a['hit@1']}/{a['n']:<4} {a['hit@3']}/{a['n']:<4} "
              f"{a['hit@5']}/{a['n']:<4} {a['mrr']:>6} {a['paper@1']}/{a['n']:<6}")
    print("\nPer-query first-hit rank (lower is better; None = gold page never retrieved in top-10):")
    for model, r in report.items():
        row = "  ".join(f"{q}:{r['per_query'][q]['first_hit_rank']}" for q in sorted(r["per_query"]))
        print(f"  {model:28} {row}")
    print("\nwrote score_report.json")


if __name__ == "__main__":
    main()
