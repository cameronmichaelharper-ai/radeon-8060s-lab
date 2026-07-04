# Vision-Native Retrieval PoC — Final Results (Marathon, 2026-07-03/04)

End-to-end test of ColPali-style page-image retrieval as an alternative/augment to
research-mcp's text-extraction+chunking pipeline, for clinical **results-table** QA.
Built and run natively on Marathon (Ryzen AI MAX+ 395, Radeon 8060S, 128GB), **CPU-only**.

## Bottom line: CONDITIONAL NO-GO (effectively no-go in the tested configuration)

Do **not** build ColPali vision-native retrieval as the primary retriever *or* as a
within-paper page reranker for research-mcp on the strength of this PoC. Instead **ship
the pipeline that already works: text-retrieve the right paper → feed its results/figure
pages to a vision reader.** Vision retrieval stays shelved, revivable only if a specific
legibility confound (below) is resolved and shown to matter.

This is a decision by **cost/schedule asymmetry + one directional failure**, NOT a powered
measurement. See caveats.

## What was built (all working, reusable)

Pipeline **fetch → render → encode → MaxSim-index → retrieve → read**, no text extraction
in the retrieval path.
- Env: uv venv Python 3.12, `torch 2.8.0+cpu` (CUDA=False by design), colpali-engine 0.3.17,
  qdrant-client 1.18.0. 3 PMC OA papers (CC BY) via Europe PMC `?pdf=render`, 51 pages @150 DPI.
- Qdrant multivector COSINE + `MAX_SIM`; proven **identical** ranking to colpali-engine's own
  `score_multi_vector` (all models are dim-128, so the parity carries across models).

## Ground truth (workflow: 3 readers + adversarial verifiers; verifier corrected q2)

| Q | paper | gold pages (0-based) | answer |
|---|---|---|---|
| q1 | PMC11153305 | 7, 4 | HbA1c effect −3.2 mmol/mol (95% CI −6.2,−0.2) p=0.04; MES −0.3 (−0.4,−0.2) p<0.001 (Table 3) |
| q2 | PMC10867699 | 4, 7, 8 | pooled SBP −2.74 mmHg (95% CI −6.43,0.95) (Fig 2/3 forest plots) |
| q3 | PMC11271409 | 0, 4 | TG −0.13 mmol/l, P=0.016 (abstract + results) |
| q4 | PMC11153305 | 5 | 100 randomized (51 FMD / 49 control) (CONSORT Fig 1) |
| q5 | PMC11271409 | 0, 4 | 4 adverse events, none severe |

## Retrieval scores vs gold (global, top-10)

| Model | params | hit@1 | hit@3 | hit@5 | MRR | paper@1 |
|---|---|---|---|---|---|---|
| colSmol-256M | 256M | 1/5 | 1/5 | 2/5 | 0.334 | 5/5 |
| colSmol-500M | 500M | 2/5 | 2/5 | 2/5 | 0.453 | 5/5 |
| colqwen2.5-v0.2 | ~3B | 2/5 | 2/5 | 3/5 | 0.494 | 5/5 |

Per-query first-gold-rank: q1 stays **8–10** in every model; q3/q5 (abstract-answerable) = rank 1.

## The decisive probe — within-paper reranking (candidates restricted to correct paper)

Because paper@1 = 5/5, the only surviving GO case was "use it as a within-paper page reranker."
Restricting candidates to the gold paper's pages:

| Model | q1 (of 15) | q2 (of 17) | q3 (of 19) | q4 (of 15) | q5 (of 19) | within-hit@3 |
|---|---|---|---|---|---|---|
| colSmol-256M | 9 | 4 | 6 | 5 | 1 | 1/5 |
| colSmol-500M | 10 | 6 | 1 | 9 | 1 | 2/5 |
| colqwen2.5-3B | 8 | 5 | 1 | 4 | 1 | 2/5 |

**The results-table page ranks LOW even among only its own paper's pages** (q1: 8–10 of 15).
The failure is not cross-paper confusion — ColPali genuinely prefers abstract/intro pages over
dense numeric tables at every scale tested. The reranker GO case is not revived.

## Reading half (downstream) — SOLVED

Lemonade vision Gemma-4-26B (mmproj), fed the gold page, returned exact matches:
- q1 Table 3 → HbA1c −3.2 (95% CI −6.2,−0.2) p=0.04; MES −0.3 (−0.4,−0.2) p<0.001 ✓
- q3 abstract → TG −0.13 mmol/l, P=0.016 ✓

So **reading is not the bottleneck; retrieval is.** If a page is retrieved, the reader nails it.
(Gemma-4 is a reasoning model — needs generous max_tokens; answer may arrive in `reasoning_content`.)

## Caveats (from adversarial review — the recommendation must not be overstated)

1. **Underpowered.** n=5 queries / 3 papers. "hit@1 1/5→2/5" is one query flipping; "scaling 12×
   barely moved it" is *unfalsifiable* at this n. The no-go rests on schedule asymmetry + q1's
   directional failure, not a powered comparison.
2. **colqwen-3B ran off-distribution.** CPU float32 + eager attn + the Qwen2.5-VL processor's
   smart-resize may have downsampled 150-DPI pages below table-cell legibility. "Bigger doesn't
   help" is therefore *unfalsified, not falsified* — the 3B result may reflect illegible input,
   not retrieval inability. (Gemma reading the same page ≠ the ColPali tower saw legible digits.)
3. **Citadel's 2/9→8/9 was answer-accuracy on a different corpus** — a reasonable prior, NOT a
   head-to-head with these retrieval metrics.
4. **The text incumbent was never run on this 51-page corpus** — this is a decision by asymmetry,
   not a measured incumbent-vs-vision comparison.
5. Only q1 is a clean deep-table query; gold is single-reader+one-verifier; ranks censored at top-10.

## Recommendation

- **Ship now:** text-route to the right paper (already reliable) → vision-read its results/figure
  pages with Lemonade Gemma (already exact). This is a defensible clinical results-table QA path on
  the critical line to Aug 2026, with zero dependence on the unresolved vision-retrieval question.
- **Do not** spend the 2.5–4 week vision-retrieval build now.
- **One cheap follow-up if ever revisited:** re-encode q1's paper at higher DPI with the Qwen2.5-VL
  processor's max_pixels raised (and, if a GPU is available, bf16 + FlashAttention-2), and re-check
  q1/q4 within-paper rank. That isolates the legibility confound — the only thing that could flip
  this. Until then, treat vision-native retrieval as shelved.

## UPDATE 2026-07-04 — read-to-rank (vision MoE as page reranker) beats ColPali on table queries

Tested Gemma-4-26B-A4B (vision MoE) as a *listwise reranker*: read each page of the routed
paper, score relevance 0–100, extract the value inline. Within-paper gold-page rank, head-to-head:

| Query | read-to-rank | colqwen2.5-3B | note |
|---|---|---|---|
| q1 HbA1c table | **2** | 8 | Table 3 (p7) scored 100 and extracted −3.2 correctly; tied with abstract (p0) |
| q2 pooled SBP | **2** | 5 | forest-plot pages p4/p7/p8 all scored 100 |
| q4 CONSORT | 15* | 4 | *artifact: gold p5 got a parse failure (no clean SCORE line); model DID extract "100 randomized, 51/49" on p0/p4/p6 |

Read-to-rank surfaces the actual results table/figure far better (q1 8→2, q2 5→2) AND returns the
extracted value in the same pass (retrieve+read fused). By "correct answer in top-3" it is ~3/3 vs
ColPali ~2/5. Wrinkles to engineer around: (a) the abstract (p0) scores 100 on every query — correct
for clinical Q&A but means the specific table isn't uniquely #1; (b) reasoning-model output fragility
— ~4/47 pages emitted no parseable SCORE (fix: JSON/grammar-constrained output or retry-on-parse-fail);
(c) many pages tie at 100 → need a finer scale or a tiebreak. **Conclusion:** vision-MoE-reader as a
within-paper reranker is the right use of Marathon and validated here; ColPali *embedding* retrieval
stays not-worth-it. Cost: 47 vision calls for 3 queries — fine at this corpus scale, not at scale.

## Files
- Scripts: `01_fetch_and_render.py` · `02_encode_and_index.py` · `03_query.py` · `04_eval_queries.py`
  · `05_sweep.py` (model-parameterized) · `06_score.py` (vs gold) · `07_read_page.py` (Lemonade vision)
  · `08_within_paper.py` (the decisive probe)
- Data: `queries.json` · `gold_labels.json` · `results_<model>.json` · `score_report.json`
  · `within_paper_report.json` · `embeddings_<model>.pkl` · `pdfs/` · `pages/<pmcid>/*.png`
- Qdrant collection `vision_pages` (validated; container has no `--restart` — `docker start qdrant`)
