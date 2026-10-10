#!/usr/bin/env bash
# 辅助 SFT 数据 (forward / yes_or_no / hld / vm_mcq / llava_video_50k) 生成流程, 在 data_gen/ 目录下运行:
#   bash scripts/run_ins_following.sh
# forward / yes_or_no / hld 依赖 run_stream_thought.sh 产出的视频切片和 KG (这里用 LLaVA-Video 那一份).
set -euo pipefail
cd "$(dirname "$0")/.."

export VST_LLM_API_URL="${VST_LLM_API_URL:?set your LLM gateway url}"
export VST_LLM_API_KEY="${VST_LLM_API_KEY:?set your LLM gateway key}"

CLIPS=videos/filted_video_clip            # <CLIPS>/<video_name>/part_{i}.mp4 (tools/split_clips.py)
KG=mhqa/data/data_kg_llava                # <KG>/<video_name>/kg_gemini3f_<video_name>.json (stage_1_generate_qa.py)
export NAME2PATH_JSON=data_2512/llava_video_name_to_path_index.json   # tools/build_name2path.py
OUT=data/curated_sft_streamvlm_format
SOTA=mhqa/data/sota_data

# 1. 基于 KG 生成三类指令跟随数据
python ins_following/forward.py    --p1_dir $CLIPS --p2_dir $KG --output_dir $SOTA/forward    --is_early_stop
python ins_following/yes_or_no.py  --p1_dir $CLIPS --p2_dir $KG --output_dir $SOTA/yes_or_no  --is_early_stop
python ins_following/hld.py        --p1_dir $CLIPS --p2_dir $KG --output_dir $SOTA/hld        --is_early_stop

# 2. 转成 StreamVLM SFT 格式
INPUT_DIR=$SOTA/forward   OUTPUT_FILE=$OUT/forward_2k.jsonl    python ins_following/format_forward.py
INPUT_DIR=$SOTA/yes_or_no OUTPUT_FILE=$OUT/yes_or_no_3k.jsonl  python ins_following/format_forward.py
INPUT_DIR=$SOTA/hld       OUTPUT_FILE=$OUT/hld_1k.jsonl        python ins_following/format_hld.py
python ins_following/format_vm_mcq.py        # VideoMarathon MCQ (data_2512/video_marathon_multi_choice.json)
python ins_following/format_llava_video.py   # 随机采样 50K LLaVA-Video 开放式 QA (data_2512/llava_video_open_end.json)

# 3. 划分 train/valid + seek 索引
for NAME in forward_2k yes_or_no_3k hld_1k vm_mcq_1k_hld llava_video_50k; do
    python tools/split_valid.py --input_path $OUT/$NAME.jsonl --output_dir $OUT --suffix with_seeks
    python tools/create_seek.py --input_path $OUT/${NAME}_train_with_seeks.jsonl
    python tools/create_seek.py --input_path $OUT/${NAME}_valid_with_seeks.jsonl
done
