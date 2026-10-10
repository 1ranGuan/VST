import json
import os
import random
import glob
from tqdm import tqdm

# ================= 配置区域 =================

# 输入文件夹路径 (请修改为你存放这些包含正负样本JSON的文件夹路径)
# yes_or_no.py 与 forward.py 的输出格式相同, 都用本脚本转换:
#   INPUT_DIR=mhqa/data/sota_data/yes_or_no OUTPUT_FILE=.../yes_or_no_3k.jsonl
#   INPUT_DIR=mhqa/data/sota_data/forward   OUTPUT_FILE=.../forward_2k.jsonl
INPUT_DIR = os.environ.get("INPUT_DIR", "mhqa/data/sota_data/yes_or_no")

# 输出文件路径
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "data/curated_sft_streamvlm_format/yes_or_no_3k.jsonl")

# 视频名称到路径的映射文件
NAME2PATH_JSON = os.environ.get("NAME2PATH_JSON", "data_2512/llava_video_name_to_path_index.json")

# System Prompt
SYSTEM_PROMPT = "[System]\nYou are a Streaming Video Analyst.\n"

# ===========================================

def load_name2path(json_path):
    """加载视频路径映射"""
    if not os.path.exists(json_path):
        env_path = os.environ.get("NAME2PATH_JSON", '')
        if env_path and os.path.exists(env_path):
            json_path = env_path
        else:
            print(f"Warning: Mapping file not found at {json_path}")
            return {}
            
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading mapping file: {e}")
        return {}

def format_yes_no_direct(qa_item):
    """
    格式化为直接回答 Yes/No 的形式，不使用 A/B 选项
    """
    question_text = qa_item.get("question", "")
    answer_text = qa_item.get("answer", "") # "Yes" or "No"
    
    # 固定显示选项列表，提示模型只能输出这两个
    options_display_str = "['Yes', 'No']"
    
    # 按照你的要求构造 final_question
    final_question = f"""{question_text}\nAnswer only with "Yes" or "No".\n"""
    
    return final_question, answer_text

def process_binary_files():
    # 1. 加载映射
    name2path = load_name2path(NAME2PATH_JSON)
    
    # 2. 获取文件列表
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory {INPUT_DIR} does not exist.")
        return

    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    print(f"Found {len(json_files)} files in {INPUT_DIR}")
    
    results = []
    
    # 3. 遍历处理
    for file_path in tqdm(json_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            video_name = data.get("video_name", "")
            video_path = name2path.get(video_name, f"MISSING_PATH_FOR_{video_name}")
            
            meta_start = data.get("start_time", 0.0)
            meta_end = data.get("end_time", 0.0)
            
            qa_list = data.get("qa", [])
            if not qa_list:
                continue
            
            # === 核心逻辑：随机选取一条 (正样本或负样本) ===
            selected_qa = random.choice(qa_list)
            
            # 格式化问题和答案
            q_str, a_str = format_yes_no_direct(selected_qa)
            
            # 获取该问题的具体时间戳
            qa_start = selected_qa.get("start_time", meta_start)
            qa_end = selected_qa.get("end_time", meta_end)
            
            # 构造目标对象
            entry = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "video_start": meta_start,
                            "video_end": meta_end
                        },
                        {
                            "type": "text",
                            "text": "",
                            "previous": SYSTEM_PROMPT,
                            "title": "",
                            "category": None
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text_steam",
                            "text_stream": [],
                            "qa_stream": [
                                [qa_start, qa_end, q_str, a_str]
                            ]
                        }
                    ]
                }
            ]
            results.append(entry)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # 4. 写入 JSONL
    print(f"Writing {len(results)} items to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for item in results:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("Done.")

if __name__ == "__main__":
    process_binary_files()