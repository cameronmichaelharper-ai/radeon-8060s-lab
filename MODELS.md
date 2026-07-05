# Model Catalog — candidates for the Radeon 8060S / Strix Halo lab

Running list of models worth testing on this machine, by role in the research-mcp pipeline
(retrieve → read → answer, plus doc-parsing for the text path). **Status** legend:
✅ tested · 🔄 running/queued · 🔎 candidate (not yet run) · 📚 reference/educational only · 👀 watch (research-stage).

## Key principle: MoE vs dense, by role

- **Generation / reading (streaming tokens)** is *bandwidth-bound* on Strix Halo → **MoE wins**
  (activate few params/token). GPT-OSS-120B ≈ 53 t/s, Qwen3-30B-A3B ≈ 86 t/s here.
- **Retrieval / embedding encoding** is a *one-time batch* job → throughput barely matters, so a
  **dense 3–8B retriever is fine**. Don't restrict the retriever search to MoE; pick the strongest.
- **Watch the Vulkan hybrid-attention bug** (see [hardware.md](hardware.md)) — it breaks several
  otherwise-attractive MoE models under llama.cpp Vulkan.

---

## 1. Visual document retrievers (late-interaction / multi-vector) — the step that failed

| Model | Arch | Params | Runs via | Status | Notes |
|---|---|---|---|---|---|
| vidore/colSmol-256M | dense (SmolVLM) | 256M | colpali-engine (`ColIdefics3`) | ✅ | weak: hit@1 1/5, buries tables |
| vidore/colSmol-500M | dense | 500M | `ColIdefics3` | ✅ | marginal: hit@1 2/5 |
| vidore/colqwen2.5-v0.2 | dense (Qwen2.5-VL) | ~3B | `ColQwen2_5` | ✅ | hit@1 2/5; table page still rank 8 |
| **nomic-ai/colnomic-embed-multimodal-3b** | dense (Qwen2.5-VL) | ~3B | `ColQwen2_5` (drop-in) | ✅ | **best pure retriever tested** — hit@5 4/5, MRR 0.52; but FMD table (q1) still rank 9 |
| nomic-ai/colnomic-embed-multimodal-7b | dense | ~7B | `ColQwen2_5` | 🔎 | stronger sibling |
| jina-embeddings-v4 | dense (Qwen2.5-VL) | 3.8B | jina/transformers | 🔎 | multi-vector **90.17 ViDoRe** (vs ColPali ~84) |
| NVIDIA Nemotron ColEmbed V2 | dense | 8B | transformers | 🔎 | **#1 ViDoRe V3** (Feb 2026), NDCG@10 63.42 |
| Argus-Retriever | **MoE** (query-conditioned) | research | — | 👀 | genuine MoE VDR retriever (arxiv 2606.04300); check for weights |

## 2. Vision-language readers / rerankers / answerers — the "read the page" step

| Model | Arch | Params (active) | Runs via | Status | Notes |
|---|---|---|---|---|---|
| **Gemma-4-26B-A4B-it** | **MoE** + vision | 26B (~4B) | Lemonade GGUF (mmproj) | ✅ | read table exactly; **beat ColPali as reranker**; already local |
| DeepSeek-VL2-small | **MoE** | 16B (2.8B) | transformers / GGUF* | 🔎 | DocVQA 93.3 (>GPT-4o); OCR/table/chart focus |
| DeepSeek-VL2-tiny / base | **MoE** | (1.0B / 4.5B) | transformers / GGUF* | 🔎 | tiny=fast, base=strongest |
| Kimi-VL-A3B (+Thinking) | **MoE** | 16B (2.8B) | transformers | 🔎 | long-CoT reasoning VLM |
| Qwen3-VL (Instruct/Thinking) | dense **and MoE** | 2B → 235B | Lemonade GGUF (Qwen3-VL-*-GGUF) | 🔎 | MoE variants (e.g. 30B-A3B-class) ideal for Strix Halo |
| Qwen2.5-VL-72B-Instruct | dense | 72B | GGUF (Mungert) | 🔎 | strong but **slow on iGPU** (dense 72B ≈ few t/s); ok for a few pages |
| Qwen2.5-VL-7B / Qwen3-VL-8B | dense | 7–8B | Lemonade GGUF | 🔎 | lighter reader baseline |
| medgemma-4b / medgemma1.5-4b | dense + vision | 4B | Lemonade (FLM/NPU) | 🔎 | **medical** VLM — relevant for FNP use |
| Astrea | **MoE** VLM | research | — | 👀 | progressive-alignment MoE VLM (arxiv 2503.09445); check weights |

\* DeepSeek-VL2 llama.cpp/GGUF support has historically been partial — verify before relying on Lemonade.

## 3. Text embedders — upgrade for the shippable text path (nomic-v1 was the bottleneck)

| Model | Arch | Params (active) | Runs via | Status | Notes |
|---|---|---|---|---|---|
| nomic-embed-text-v1 | dense | 137M | Lemonade GGUF | ✅ (incumbent) | Citadel eval flagged as bottleneck |
| **nomic-embed-text-v2-moe** | **MoE** | 475M (305M) | Lemonade GGUF (**in catalog**) | 🔎 | direct drop-in upgrade; Matryoshka 768→256 |
| BGE-M3 | dense | 568M | Lemonade/GGUF | 🔎 | strong multilingual hybrid (dense+sparse+colbert) |
| Qwen3-Embedding 0.6/4/8B | dense | — | Lemonade GGUF (**in catalog**) | 🔎 | flexible dim; 8B strong |

## 4. Document parsers — structured extraction for the text path (the *original* problem)

| Model | Arch | Params | Runs via | Status | Notes |
|---|---|---|---|---|---|
| NVIDIA Nemotron Parse 1.1 | encoder-decoder (RADIO ViT-H + mBART) | ~900M | transformers | 🔎 | lightweight; layout + **tables** + bboxes + reading order. TC variant ~20% faster. Feeds better chunks than pymupdf |
| RAGFlow deepdoc | pipeline | — | container (image kept) | 📚 | layout parser already on disk; standalone/offline |
| pymupdf4llm | lib | — | pip | 🔎 | markdown+table extraction (Citadel cheap-win) |

## 5. Rerankers (for the text path)

| Model | Runs via | Status | Notes |
|---|---|---|---|
| bge-reranker-v2-m3 | Lemonade GGUF (**in catalog**) | 🔎 | Citadel cheap-win #3; start Lemonade with `--reranking` |
| jina-reranker-v1-tiny-en | Lemonade GGUF (**in catalog**) | 🔎 | tiny/fast |

## 6. Frameworks / educational (not deployable models)

| Name | What | Use |
|---|---|---|
| SeeMoE (AviSoori1x) | 📚 build a MoE VLM from scratch in PyTorch (scaled Grok-1.5-V/GPT-4V) | learning / train-your-own reference |
| makeMoE | 📚 build a text MoE from scratch | learning reference |

---

### Tested-so-far summary (experiment 01) — powered 14-query eval

| Model | hit@1 | hit@5 | MRR | verdict |
|---|---|---|---|---|
| colSmol-256M | 3/14 | 8/14 | 0.39 | weakest |
| colSmol-500M | **7/14** | 9/14 | **0.58** | best hit@1 (surprise) |
| colqwen2.5-v0.2 | 5/14 | 10/14 | 0.50 | middling |
| colnomic-embed-multimodal-3b | 6/14 | **11/14** | 0.58 | **best recall** (feeds reranker) |
| Gemma-4-26B-A4B (read-to-rank) | best on table queries (n=3) | — | — | **best table localization** |

**Takeaway (updated):** on 14 queries the retrievers are closer than n=5 suggested — 500M ties/leads
on hit@1, colnomic leads on recall (79% hit@5). BUT the cheap **colnomic top-10 → Gemma rerank hybrid
tested at hit@1 2/5 = pure colnomic (no gain)**: the specific table page is often outside top-10, the
reader over-scores look-alikes, and cross-paper candidates contaminate. *Reading* is solved; cheap
retrieval *of the exact table page* is the open problem. Next levers: top-20 + restrict-to-routed-paper
+ tie-break pass, or a stronger retriever (jina-embeddings-v4 multi-vector, Nemotron ColEmbed 8B).
