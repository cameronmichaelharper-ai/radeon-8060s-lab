"""09_read_to_rank.py -- read-to-rank: use the vision MoE (Lemonade Gemma-4-26B-A4B)
as a listwise page reranker over the ROUTED paper's pages, instead of ColPali embeddings.

Since paper@1 = 5/5 (routing solved), for each query we only score the gold paper's
pages. The model READS each page and scores how directly it answers the query. We then
rank pages by that score and report the rank of the gold page -- the exact metric
08_within_paper.py produced for ColPali, so it's a head-to-head.

Default queries = the 3 table/figure-dependent ones ColPali buried (q1, q2, q4).
"""
import os
import re
import json
import glob
import base64
import urllib.request

WORKDIR = os.path.dirname(os.path.abspath(__file__))
PAGES_DIR = os.path.join(WORKDIR, "pages")
LEMONADE = "http://localhost:13305/api/v1/chat/completions"
MODEL = "Gemma-4-26B-A4B-it-GGUF"
QIDS = os.environ.get("QIDS", "q1,q2,q4").split(",")

SCORE_RE = re.compile(r"SCORE:\s*(\d{1,3})", re.IGNORECASE)
VALUE_RE = re.compile(r"VALUE:\s*(.+)", re.IGNORECASE)


def score_page(png_path, question):
    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    prompt = (
        "You are ranking pages of a scientific paper by how directly THIS page answers a "
        "question. Look at the page image.\n\nQuestion: " + question + "\n\n"
        "Score HIGH only if this page contains the specific answer (the results table, the "
        "figure/forest-plot, the CONSORT flow diagram, or the stated numeric value that directly "
        "answers it). Score LOW if the page only mentions the topic, or is title/abstract/intro/"
        "methods/discussion/references without the actual result.\n\n"
        "Respond with EXACTLY two lines and nothing else:\n"
        "SCORE: <integer 0-100>\nVALUE: <the specific number/result if present on this page, else NONE>"
    )
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        "temperature": 0.0, "max_tokens": 1400,
    }
    req = urllib.request.Request(LEMONADE, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        msg = json.load(r)["choices"][0]["message"]
    text = (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "")
    m = SCORE_RE.search(text)
    v = VALUE_RE.search(text)
    score = int(m.group(1)) if m else -1
    value = v.group(1).strip() if v else ""
    return score, value


def pages_of(pmcid):
    return sorted(glob.glob(os.path.join(PAGES_DIR, pmcid, "*.png")))


def main():
    queries = {q["id"]: q for q in json.load(open(os.path.join(WORKDIR, "queries.json"), encoding="utf-8"))}
    gold = {l["query_id"]: l for l in json.load(open(os.path.join(WORKDIR, "gold_labels.json"), encoding="utf-8"))["labels"]}

    report = {}
    for qid in QIDS:
        q = queries[qid]
        g = gold[qid]
        gold_pages = set(g["gold_pages"])
        pmcid = g["pmcid"]
        pngs = pages_of(pmcid)
        print(f"\n=== {qid} [{pmcid}] scoring {len(pngs)} pages ===")
        scored = []
        for png in pngs:
            idx = int(os.path.splitext(os.path.basename(png))[0].rsplit("_p", 1)[1])
            s, val = score_page(png, q["text"])
            scored.append({"page": idx, "score": s, "value": val})
            flag = " <-- GOLD" if idx in gold_pages else ""
            print(f"  p{idx:02d} score={s:>3} {('val=' + val[:60]) if val and val.upper()!='NONE' else ''}{flag}")
        ranked = sorted(scored, key=lambda x: x["score"], reverse=True)
        rank = next((r for r, x in enumerate(ranked, 1) if x["page"] in gold_pages), None)
        top = ranked[0]
        report[qid] = {"pmcid": pmcid, "gold_pages": sorted(gold_pages), "n_pages": len(pngs),
                       "read_to_rank": rank, "top_page": top["page"], "top_value": top["value"],
                       "scored": scored}
        print(f"  => gold page rank {rank} of {len(pngs)} (top page p{top['page']}, value: {top['value'][:80]})")

    json.dump(report, open(os.path.join(WORKDIR, "read_to_rank_report.json"), "w", encoding="utf-8"), indent=2)
    print("\n=== READ-TO-RANK vs ColPali (within-paper gold-page rank; lower=better) ===")
    cp = json.load(open(os.path.join(WORKDIR, "within_paper_report.json"), encoding="utf-8"))
    cq = cp.get("vidore/colqwen2.5-v0.2", {}).get("per_q", {})
    for qid in QIDS:
        r = report[qid]
        cpr = cq.get(qid, {}).get("within_rank")
        print(f"  {qid} [{r['pmcid']}]: read-to-rank={r['read_to_rank']}  |  colqwen2.5-3B={cpr}  (of {r['n_pages']} pages)")
    print("\nwrote read_to_rank_report.json")


if __name__ == "__main__":
    main()
