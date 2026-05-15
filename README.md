<div align="center">

# 🎬 Video Streaming Thinking (VST)

### VideoLLMs Can Watch and Think Simultaneously

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2603.12262)
[![Homepage](https://img.shields.io/badge/Homepage-project-orange.svg?logo=googlehome)](https://1ranguan.github.io/VST/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)]()
[![Model 3/7/32B](https://img.shields.io/badge/🤗%20Model-HuggingFace-yellow?style=flat-square)](https://huggingface.co/Catalan258/VST-7B)
[![Training Data](https://img.shields.io/badge/🤗%20Training%20Data-HuggingFace-yellow?style=flat-square)](https://huggingface.co/datasets/Catalan258/VST-Training-Data)

</div>

> **Video Streaming Thinking** introduces a new paradigm for streaming video understanding that interleaves active reasoning with continuous video consumption, enabling amortized test-time scaling with real-time responsiveness.

---

## 🔍 Overview

Existing online VideoLLMs focus on efficient streaming perception but lack explicit analytical reasoning. Offline VideoLLMs with Chain-of-Thought (CoT) can reason deeply, but incur high query-answer (QA) latency that violates real-time constraints. **VST bridges this gap** by shifting the LLM backend from passive waiting to active, intermittent reasoning *during* video consumption, implementing a **thinking-while-watching** mechanism inspired by human neural coupling.

https://github.com/user-attachments/assets/49846db5-bf76-4cf8-b923-4b9b88117482

### ✨ Key Idea

Instead of deferring all reasoning until a user query arrives, VST continuously processes incoming video clips and produces **intermediate streaming thoughts** in real time. This front-loads and amortizes the reasoning cost, so the final response is both **deeply grounded** and **instantly available**.

## 🏗️ Model Zoo

| **Model** | **HuggingFace** | **OVO-Bench** | **StreamingBench** | **VideoMME** | **LongVideoBench** | **VideoHolmes** |
|---|---|---|---|---|---|---|
| VST-3B | [🤗 Link](https://huggingface.co/Catalan258/VST-3B) | 56.2 | 75.5 | 59.5 | 54.1 | 36.1 |
| VST-7B | [🤗 Link](https://huggingface.co/Catalan258/VST-7B) | 59.3 | 79.5 | 64.9 | 58.0 | 41.9 |
| VST-32B | [🤗 Link](https://huggingface.co/Catalan258/VST-32B) | 63.5 | 80.7 | 67.2 | 60.7 | 45.1 |

## 📦 Training Data

We release the full training data used for both SFT and RL stages on HuggingFace:

| **Dataset** | **Description** |
|---|---|
| [VST-Training-Data/vst_sft_data](https://huggingface.co/datasets/Catalan258/VST-Training-Data/tree/main/vst_sft_data) | SFT data including video-text pairs from multiple sources |
| [VST-Training-Data/vst_rl_data](https://huggingface.co/datasets/Catalan258/VST-Training-Data/tree/main/vst_rl_data) | RL data for reinforcement learning stage |

## 📅 TODO
- [x] Release the paper.
- [x] Release checkpoint and eval code.
- [x] Release training code.
- [x] Release training data.

## 👍 Acknowledgement
We thank the following great works and open-source repositories:
- [StreamingVLM](https://github.com/mit-han-lab/streaming-vlm)
- [MemAgent](https://github.com/BytedTsinghua-SIA/MemAgent)
- [Streamingthinker](https://github.com/EIT-NLP/StreamingLLM)
## 📖 Citation
```
@article{guan2026videostreamingthinking,
      title={Video Streaming Thinking: VideoLLMs Can Watch and Think Simultaneously}, 
      author={Yiran Guan and Liang Yin and Dingkang Liang and Jianzhong Ju and Zhenbo Luo and Jian Luan and Yuliang Liu and Xiang Bai},
      journal={arXiv preprint arXiv:2603.12262},
      year={2026},
}
```
