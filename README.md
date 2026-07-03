# Case Studies

Hands-on engineering write-ups by **CryptoAI-Jedi**: real builds, real constraints, and the non-obvious problems solved along the way. The focus is AI infrastructure and LLM systems, with a crypto and onchain slant.

## Studies

### [Turning 4 Ex-Mining AMD GPUs into a Local LLM Inference Server](./case-study-ex-mining-amd-llm-inference.md)
Four ex-Ethereum-mining RX 5600 XT cards (RDNA1) serving a real LLM over an OpenAI-compatible API, with no CUDA and no ROCm, via Mesa RADV Vulkan and llama.cpp. Walks the three non-obvious failures behind it: a 180x throughput collapse from cross-card decode over x1 Gen1 risers, a headless GPU silently serving from system RAM instead of VRAM, and an OOM kernel panic that had nothing to do with VRAM, with the diagnostic path for each.

## Elsewhere
- **OnchainIQ** — a natural-language onchain intelligence agent, built in the open: https://github.com/CryptoAI-Jedi/OnchainIQ
- Build-in-public updates on X: **@CryptoAIJedi**
