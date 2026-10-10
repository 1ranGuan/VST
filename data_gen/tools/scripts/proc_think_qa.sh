# python tools/select_combine_llava_video_qa.py

cd "$(dirname "$0")/../.."  # run from data_gen/


LLAVA_VIDEO_QA=data_2512/llava_video_open_end.json
VIDEO_MARATHON_QA=data_2512/video_marathon_open_end.json


###########################要改的第一个地方
THINKING_FILE=data/curated_sft_streamvlm_format/Streaming_cot_17k_128s_1222_llava.jsonl
###########################要改的第二个地方
OUTPUT_FILE=data/curated_sft_streamvlm_format/Streaming_cot_17k_128s_1222_llava_QA.jsonl


#combined QA
python tools/merge_qa_thinking.py \
    --qa_files "$LLAVA_VIDEO_QA" "$VIDEO_MARATHON_QA" \
    --thinking_file "$THINKING_FILE" \
    --output_file "$OUTPUT_FILE"



INPUT_DIR=data/curated_sft_streamvlm_format
OUTPUT_DIR=data/curated_sft_streamvlm_format

###########################要改的第三个地方
INPUT_FILE_NAME=Streaming_cot_17k_128s_1222_llava_QA


echo $INPUT_DIR 
echo $INPUT_FILE_NAME 

cp  "${INPUT_DIR}/${INPUT_FILE_NAME}.jsonl" \
    ${OUTPUT_DIR}


#split train and valid set
python tools/split_valid.py \
    --input_path "${INPUT_DIR}/${INPUT_FILE_NAME}.jsonl" \
    --output_dir ${OUTPUT_DIR} \
    --suffix with_seeks 

#create seeks for train and valid set
TRAIN_PATH="${OUTPUT_DIR}/${INPUT_FILE_NAME}_train_with_seeks.jsonl"
VALID_PATH="${OUTPUT_DIR}/${INPUT_FILE_NAME}_valid_with_seeks.jsonl"

python tools/create_seek.py \
    --input_path $TRAIN_PATH

python tools/create_seek.py \
    --input_path $VALID_PATH