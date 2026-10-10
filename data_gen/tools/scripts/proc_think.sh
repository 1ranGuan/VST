# python tools/select_combine_llava_video_qa.py

cd "$(dirname "$0")/../.."  # run from data_gen/





######################需要改的参数
INPUT_DIR=data/curated_sft_streamvlm_format
OUTPUT_DIR=data/curated_sft_streamvlm_format
INPUT_FILE_NAME=vm_mcq_1k_hld
######################需要改的参数

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