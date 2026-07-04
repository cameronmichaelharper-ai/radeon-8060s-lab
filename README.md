# Radeon 8060S Local-AI Lab

Cutting-edge local AI experiments on an **AMD Radeon 8060S iGPU** — the integrated GPU of the
**Ryzen AI MAX+ 395 "Strix Halo"** (gfx1151), 128 GB unified memory, run from a home office.
Searchable aliases: *Strix Halo*, *Ryzen AI Max 395*, *gfx1151*, *Radeon 8060S*, *AMD APU LLM*,
*Lemonade Server*, *llama.cpp Vulkan*, *ROCm 7 gfx1151*.

This corner of the hardware world is under-documented — most local-LLM tooling assumes NVIDIA/CUDA.
The goal here is to publish reproducible results (what runs, how fast, what breaks) so others
searching these topics find real data instead of guesses.

## Why this exists

The AMD path (Vulkan / ROCm / DirectML, no CUDA) has less community optimization and far fewer
worked examples than NVIDIA. Everything here is run natively on the machine, with the exact
commands, versions, and numbers, including the failures and the workarounds.

## Experiments

| # | Title | Status | One-line finding |
|---|---|---|---|
| 01 | [Vision-native retrieval (ColPali) vs text pipeline](experiments/01-vision-native-retrieval/RESULTS.md) | done + extending | ColPali embedding retrieval buries results tables at all model sizes; **vision-MoE-as-reader-reranker** (Gemma-4-26B-A4B) beats it and reads the value inline. colnomic-3b retest in progress. |

## Reference docs

- [MODELS.md](MODELS.md) — running catalog of candidate models to test (retrievers, vision readers,
  text embedders, doc parsers, frameworks), with architecture, how-to-run-on-Strix-Halo, and status.
- [hardware.md](hardware.md) — the machine profile, the local model stack (Lemonade), measured
  MoE throughput, and the known Vulkan hybrid-attention bug list.

## How to reproduce

Each experiment folder holds its own scripts and a `RESULTS.md`. Large/regenerable artifacts
(venvs, model weights, rendered pages, embeddings, vector-store data) are gitignored — rebuild
from `ENV_SETUP.txt` + the numbered scripts. Everything runs CPU or Vulkan-iGPU; no CUDA.

*Maintained by Cameron Harper. Machine name "Marathon."*
