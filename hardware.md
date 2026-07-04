# Hardware & local stack — "Marathon"

## Machine
- **ASUS ROG Flow Z13 (GZ302EAC)**
- **APU:** AMD Ryzen AI MAX+ 395 "Strix Halo"
- **iGPU:** Radeon 8060S (RDNA 3.5, **gfx1151**)
- **NPU:** XDNA2 (Ryzen AI)
- **Memory:** 128 GB unified LPDDR5X (VGM-configurable; ~96 GB can be dedicated as VRAM)
- **No discrete GPU, no CUDA.** Compute paths: **Vulkan** (llama.cpp) and **ROCm 7.x** (gfx1151);
  DirectML/ONNX as fallback. **WSL2 + ROCm PyTorch is a known dead end on gfx1151** — use native.
- OS: Windows 11. Tailscale `100.113.59.116`.

## Local inference stack
- **Lemonade Server** v10.8.0 — OpenAI-compatible, `http://localhost:13305`.
  - Endpoints: `/api/v1/models`, `/chat/completions`, `/embeddings` (works), `/reranking`
    (needs server started with `--reranking`).
  - Gotcha: ships no `-ngl` by default → CPU fallback ~20 t/s in shared-mem mode.
    Fix: `lemonade config set enable_dgpu_gtt=true` (auto GTT offload). Local GGUFs via
    `extra_models_dir` (the pull API rejects local paths).
- **llama.cpp** via Vulkan (AMDVLK / RADV) for token-gen; ROCm 7.x nightlies for prefill/batch.
- uv-managed Python venvs pinned to 3.12 (system Python 3.14 is too new for ML wheels).

## Measured MoE throughput (Vulkan, this class of machine, 2026)
| Model | Total / active | t/s |
|---|---|---|
| Qwen3-30B-A3B | 30B / 3B | ~86 |
| Qwen3-Coder-30B-A3B | 30B / 3B | 74.4 (our measure) |
| Qwen3-Next-80B-A3B | 80B / 3B | ~59 |
| GPT-OSS-120B | 120B / ~5B | 42–53 |
| gpt-oss-120b **needs `--no-mmap`** | | 3× faster load, frees RAM |

**Rule:** on a bandwidth-bound APU, tokens/s tracks *active* params, not total — prioritize MoE.

## Known Vulkan hybrid-attention bug (breaks these under Vulkan)
- Qwen3-Next-80B-A3B (drops to ~16 t/s)
- Qwen3.5-122B-A10B
- GLM-4.7-Flash
Check ROCm path or a llama.cpp build with the fix before relying on these.

## Downloaded / in Lemonade catalog (relevant)
Vision: **Gemma-4-26B-A4B-it (MoE, mmproj)**, Qwen3-VL-4B/8B, Qwen2.5-VL-3B/7B, medgemma-4b (NPU).
Embed: nomic-embed-text-v1 (local), nomic-embed-text-v2-moe, Qwen3-Embedding-0.6/4/8B.
Rerank: bge-reranker-v2-m3, jina-reranker-v1-tiny-en.
Text MoE: gpt-oss-120b, Qwen3-Coder-30B-A3B, Qwen3.6-35B-A3B.
