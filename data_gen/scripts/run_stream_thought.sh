#!/usr/bin/env bash
# 流式思考数据 (streamvlm_stage2_*) 生成流程, 在 data_gen/ 目录下运行:
#   bash scripts/run_stream_thought.sh
# 依次: 视频切片 -> KG 构建 + 多跳 QA -> 逐段流式思考 -> 拼成 SFT 格式 -> 划分 train/valid + seek 索引
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- 需要自己设置 ----
# Vertex AI (KG 构建 + QA 生成, mhqa/osvGraph.py / mhqa/qaGenerator.py)
export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:?set to your GCP service-account json}"
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:?set your GCP project id}"
# HTTP LLM 网关 (流式思考生成, mhqa/stage_3_add_discription.py), 请求格式见 mhqa/utils.py:call_llm_api
export VST_LLM_API_URL="${VST_LLM_API_URL:?set your LLM gateway url}"
export VST_LLM_API_KEY="${VST_LLM_API_KEY:?set your LLM gateway key}"

ANNO_JSON="data_2512/video_marathon_multi_choice.json data_2512/video_marathon_open_end.json"  # 原始 QA 标注 (含 video/duration/id)
DATASET_PATH="${DATASET_PATH:?set root of the raw videos}"                                   # 标注里 video 字段相对它
WORK=mhqa/data
NAME=streamvlm_stage2_vm
# ---------------------

export VIDEO_PATH=videos/vm_video_clip            # 切片输出: <VIDEO_PATH>/<video_name>/part_{i}.mp4
export SAVE_PATH=$WORK/data_kg                    # KG: <SAVE_PATH>/<video_name>/kg_gemini3f_<video_name>.json
export OUTPUT_FOLDER=$WORK/qa_dataset_step1       # 多跳 QA
export EXISTING_PROCESSED_FILE=process_vm/qa_gen_processed_list.json
export FILTED_VM_1224=$WORK/video_folders.json    # 要处理的视频目录名列表
export GRAPH_LOG_ROOT=process_vm/graph_logs      # 每个视频的运行日志
mkdir -p "$GRAPH_LOG_ROOT"

# 0. 视频切片 (按时长等分 8/16/32 段) + 视频名 -> 原路径索引
python tools/split_clips.py --anno_json $ANNO_JSON --video_root "$DATASET_PATH" --output_root "$VIDEO_PATH"
python tools/build_name2path.py --anno_json $ANNO_JSON --output $WORK/name2path.json
mkdir -p $WORK && python -c "import json,os; json.dump(sorted(os.listdir('$VIDEO_PATH')), open('$FILTED_VM_1224','w'))"

# 1. 流式 KG 构建 + 证据链采样 + 多跳 QA 生成 (Gemini, Vertex AI)
python mhqa/stage_1_generate_qa.py   # 脚本目录 mhqa/ 自动进 sys.path

# 2. 逐段生成流式思考 (增量描述, 不看问题)
python mhqa/stage_3_add_discription.py --p1_dir "$VIDEO_PATH" --p2_dir "$SAVE_PATH" --output_dir $WORK/captions --is_early_stop

# 3. 流式思考 + QA 拼成 StreamVLM SFT 格式
CAPTION_ROOT=$WORK/captions QA_ROOT=$OUTPUT_FOLDER NAME2PATH_JSON=$WORK/name2path.json \
OUTPUT_FILE=data/curated_sft_streamvlm_format/$NAME.jsonl python mhqa/collect_format.py

# 4. 划分 train/valid 并生成 VST-SFT 读取用的 *_seeks.jsonl 索引
OUT=data/curated_sft_streamvlm_format
python tools/split_valid.py --input_path $OUT/$NAME.jsonl --output_dir $OUT --suffix with_seeks
python tools/create_seek.py --input_path $OUT/${NAME}_train_with_seeks.jsonl
python tools/create_seek.py --input_path $OUT/${NAME}_valid_with_seeks.jsonl
