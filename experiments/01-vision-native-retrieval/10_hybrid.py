"""10_hybrid.py -- the recommended architecture, end to end.

Stage 1 (cheap, dense): colnomic retrieves top-K candidate pages (read from the
existing results_colnomic-*.json, so no re-encoding).
Stage 2 (expensive, precise): vision-MoE (Lemonade Gemma-4-26B) read-to-ranks ONLY
those K candidates -- reads each page, scores relevance, extracts the value.
Final pick = Gemma's top page. Score vs gold.

Reports, per query: whether the gold page was even in colnomic's top-K (recall@K --
the retriever's ceiling), and whether the hybrid's final #1 is a gold page (hybrid hit@1).
K defaults to 10 because colnomic buried q1's table at rank 9 (top-5 would miss it).

Gemma calls are LOCAL (Lemonade) -- no cloud budget.
"""
import os
import re
import time
import json
import base64
import urllib.request

WORKDIR = os.path.dirname(os.path.abspath(__file__))
PAGES_DIR = os.path.join(WORKDIR, "pages")
RETRIEVER = os.environ.get("RETRIEVER", "colnomic-embed-multimodal-3b")
RESULTS = os.path.join(WORKDIR, f"results_{RETRIEVER}.json")
QUERIES_FILE = os.environ.get("QUERIES_FILE", os.path.join(WORKDIR, "queries.json"))
GOLD_FILE = os.environ.get("GOLD_FILE", os.path.join(WORKDIR, "gold_labels.json"))
K = int(os.environ.get("K_RETRIEVE", "10"))
LEMONADE = "http://localhost:13305/api/v1/chat/completions"
GEMMA = "Gemma-4-26B-A4B-it-GGUF"
SCORE_RE = re.compile(r"SCORE:\s*(\d{1,3})", re.IGNORECASE)
VALUE_RE = re.compile(r"VALUE:\s*(.+)", re.IGNORECASE)


def gemma_score(png_path, question):
    """Return (score 0-100, value). Retries on HTTP/parse failure; neutral 50 on give-up
    (so a real gold page is not sunk to the bottom by a transient blip)."""
    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    prompt = (
        "You are ranking pages of a scientific paper to find the PRIMARY SOURCE of a specific "
        "answer.\n\nQuestion: " + question + "\n\nScore in ONE band:\n"
        "90-100: THIS page IS the results TABLE, FIGURE/forest-plot, or CONSORT diagram that "
        "contains the specific numeric answer.\n"
        "60-85: THIS page states the answer only in prose (abstract, results narrative, or "
        "discussion) - correct value but not the source table/figure.\n"
        "0-30: intro/methods/references or a page without the answer.\n\n"
        "Respond with EXACTLY two lines:\nSCORE: <integer 0-100>\nVALUE: <the number/result if present, else NONE>"
    )
    payload = {"model": GEMMA, "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]}], "temperature": 0.0, "max_tokens": 1400}
    data = json.dumps(payload).encode()
    last = "?"
    for attempt in range(3):
        try:
            req = urllib.request.Request(LEMONADE, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                msg = json.load(r)["choices"][0]["message"]
            text = (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "")
            m, v = SCORE_RE.search(text), VALUE_RE.search(text)
            if m:
                return int(m.group(1)), (v.group(1).strip() if v else "")
            last = "no SCORE line parsed"
        except Exception as e:  # noqa: BLE001 -- transient HTTP 500 / timeout
            last = str(e)
        time.sleep(2)
    return 50, f"[fallback after retries: {last}]"


def png_for(pmcid, page):
    return os.path.join(PAGES_DIR, pmcid, f"{pmcid}_p{page:03d}.png")


def main():
    results = json.load(open(RESULTS, encoding="utf-8"))["queries"]
    gold = {l["query_id"]: l for l in json.load(open(GOLD_FILE, encoding="utf-8"))["labels"]}
    qtext = {q["id"]: q["text"] for q in json.load(open(QUERIES_FILE, encoding="utf-8"))}

    allowed = {x for x in os.environ.get("QIDS", "").split(",") if x}
    report = {}
    recall_hits = hybrid_hits = n = 0
    for qid, ranked in results.items():
        if qid not in gold or (allowed and qid not in allowed):
            continue
        g = gold[qid]
        goldset = set(g["gold_pages"])
        cands = ranked[:K]
        recall = any(c["pmcid"] == g["pmcid"] and c["page"] in goldset for c in cands)

        print(f"\n=== {qid} [{g['pmcid']}] gold {sorted(goldset)} | colnomic top-{K}, recall@{K}={recall} ===")
        scored = []
        for c in cands:
            s, val = gemma_score(png_for(c["pmcid"], c["page"]), qtext[qid])
            scored.append({**c, "gemma": s, "value": val})
            gflag = " <-- GOLD" if (c["pmcid"] == g["pmcid"] and c["page"] in goldset) else ""
            print(f"  {c['pmcid']}:p{c['page']:02d} colnomic={c['score']:.1f} gemma={s:>3}{gflag}")
        reranked = sorted(scored, key=lambda x: x["gemma"], reverse=True)
        final = reranked[0]
        hybrid_hit1 = final["pmcid"] == g["pmcid"] and final["page"] in goldset
        gold_rank = next((r for r, x in enumerate(reranked, 1)
                          if x["pmcid"] == g["pmcid"] and x["page"] in goldset), None)
        n += 1
        recall_hits += int(recall)
        hybrid_hits += int(hybrid_hit1)
        report[qid] = {"pmcid": g["pmcid"], "gold_pages": sorted(goldset),
                       "recall@K": recall, "hybrid_hit@1": hybrid_hit1,
                       "gold_rank_after_rerank": gold_rank,
                       "final_page": final["page"], "final_value": final["value"]}
        print(f"  => hybrid pick p{final['page']} ({'GOLD' if hybrid_hit1 else 'miss'}); "
              f"gold rank after rerank = {gold_rank}; value: {final['value'][:80]}")

    report["_summary"] = {"K": K, "n": n, "retriever_recall@K": f"{recall_hits}/{n}",
                          "hybrid_hit@1": f"{hybrid_hits}/{n}"}
    json.dump(report, open(os.path.join(WORKDIR, "hybrid_report.json"), "w", encoding="utf-8"), indent=2)
    print(f"\n=== HYBRID SUMMARY (K={K}) ===")
    print(f"  colnomic recall@{K}: {recall_hits}/{n}  (ceiling: can the reader even see the gold page?)")
    print(f"  hybrid hit@1:        {hybrid_hits}/{n}  (vs pure colnomic hit@1 2/5)")
    print("wrote hybrid_report.json")


if __name__ == "__main__":
    main()
