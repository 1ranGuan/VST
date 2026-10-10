# VST Data Generation

Code used to synthesize the VST SFT data released at
[HF `Catalan258/VST-Training-Data`](https://huggingface.co/datasets/Catalan258/VST-Training-Data/tree/main/vst_sft_data) (`vst_sft_data/`).
It implements the streaming-thought data pipeline described in the paper: streaming entity extraction into a video knowledge graph, evidence-chain sampling, multi-hop QA synthesis, and per-clip streaming thoughts.

Run everything from this `data_gen/` directory.

```bash
pip install -r requirements.txt   # plus ffmpeg on PATH
```

## Credentials

| Variable | Used by | Purpose |
|---|---|---|
| `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_PROJECT` (`GOOGLE_CLOUD_LOCATION`, default `global`) | `mhqa/osvGraph.py`, `mhqa/qaGenerator.py` | Gemini on Vertex AI (KG + QA) |
| `VST_LLM_API_URL`, `VST_LLM_API_KEY` | `mhqa/stage_*`, `mhqa/post_filter.py`, `ins_following/*` | An HTTP LLM gateway serving Gemini. Request: `POST {model, key_id, contents, system_prompt}` with header `X-API-Key` (see `mhqa/utils.py:call_llm_api`); adapt `_call_api` in each script if your provider differs. |

## Pipeline 1: streaming-thought data (`streamvlm_stage2_*`)

`bash scripts/run_stream_thought.sh` runs these steps (edit the variables at its top first):

| Step | Script | Output |
|---|---|---|
| 0. Clip videos | `tools/split_clips.py` | `<VIDEO_PATH>/<video>/part_{i}.mp4`. Each video is split into 8 / 16 / 32 equal-length clips for durations < 256 s / < 1024 s / ≤ 2048 s; longer videos are dropped |
| 0. Name → path index | `tools/build_name2path.py` | `{video_name: dataset-relative path}` |
| 1. KG + multi-hop QA | `mhqa/stage_1_generate_qa.py` (→ `qaGenerator.py`, `osvGraph.py`) | `kg_gemini3f_<video>.json` (entity–relation triples per clip, built with a sliding window over clips) and the QA per video (question / cot / answer, sampled from KG evidence chains) |
| 2. Streaming thoughts | `mhqa/stage_3_add_discription.py` | per-clip incremental thoughts (`captions/<video>.json`) |
| 3. SFT format | `mhqa/collect_format.py` | StreamVLM-style jsonl: thoughts interleaved with clips in 128 s windows (duration/8 for videos longer than 1024 s), QA appended |
| 4. Split + index | `tools/split_valid.py`, `tools/create_seek.py` | `*_train_with_seeks.jsonl` / `*_valid_with_seeks.jsonl` + `*_seeks.jsonl` (byte offsets read by `VST-SFT/streaming_vlm/data/lmm_dataset.py`) |

Optional QA post-processing (not used for the released `streamvlm_stage2_*` files): `stage_2_filter_qa.py` (world-knowledge / grounding filter), `post_filter.py` (answerability from thoughts), `stage_mcq.py` (open-ended → 4-way MCQ), `stage_4_format_streamvlm.py` (reasoning-style SFT format).

## Pipeline 2: auxiliary SFT data

`bash scripts/run_ins_following.sh` builds `forward_2k`, `yes_or_no_3k`, `hld_1k` (generated from the clips + KG of pipeline 1 via `ins_following/{forward,yes_or_no,hld}.py`), plus `vm_mcq_1k_hld` (VideoMarathon MCQ) and `llava_video_50k` (50K LLaVA-Video open-ended QA).

## Other tools

- `tools/merge_qa_thinking.py`, `tools/scripts/proc_think*.sh`: merge external QA into thought files, then split and index them.
- `tools/convert_sft.py`, `tools/convert_eval.py`: format conversions for SFT and eval toy data.
- `mhqa/rewrite_counting_cs.py`: rewrites the counting cold-start set.
- `ins_following/statistic.py`: data-mixture statistics.
- `mhqa/tools/count_qa.py`, `mhqa/tools/filter_video.py`: QA counting and video filtering.

Model names (`gemini-3-flash-preview` / `gemini-3-flash`), worker counts, and retry settings are module-level constants at the top of each script.
