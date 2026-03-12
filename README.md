<div align="center">

# 🎬 Video Streaming Thinking (VST)

### VideoLLMs Can Watch and Think Simultaneously

### [Project Page](https://1ranguan.github.io/VST/)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?style=flat-square&logo=arxiv)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)]()

</div>


> **Video Streaming Thinking** introduces a new paradigm for streaming video understanding that interleaves active reasoning with continuous video consumption, enabling amortized test-time scaling with real-time responsiveness.

---

## 🔍 Overview

Existing online VideoLLMs focus on efficient streaming perception but lack explicit analytical reasoning. Offline VideoLLMs with Chain-of-Thought (CoT) can reason deeply, but incur high query-answer (QA) latency that violates real-time constraints. **VST bridges this gap** by shifting the LLM backend from passive waiting to active, intermittent reasoning *during* video consumption, implementing a **thinking-while-watching** mechanism inspired by human neural coupling.

### ✨ Key Idea

Instead of deferring all reasoning until a user query arrives, VST continuously processes incoming video clips and produces **intermediate streaming thoughts** in real time. This front-loads and amortizes the reasoning cost, so the final response is both **deeply grounded** and **instantly available**.

## 🏗️ Model Zoo

| **Model** | **OVO-Bench** | **StreamingBench** | **VideoMME** | **LongVideoBench** | **VideoHolmes** |
|---|---|---|---|---|---|
| VST-3B | 56.2 | 75.5 | 59.5 | 54.1 | 36.1 |
| VST-7B | 59.3 | 79.5 | 64.9 | 58.0 | 41.9 |
| VST-32B | 63.5 | 80.7 | 67.2 | 60.7 | 45.1 |

## 📅 TODO
- [x] Release the paper.
- [ ] Release checkpoint and eval code.
- [ ] Release training code.
- [ ] Release training data.


## 👍 Acknowledgement
We thank the following great works and open-source repositories:
- [StreamingVLM](https://github.com/mit-han-lab/streaming-vlm)
- [MemAgent](https://github.com/BytedTsinghua-SIA/MemAgent)
- [Streamingthinker](https://github.com/EIT-NLP/StreamingLLM)
## 📖 Citation (TBD)
