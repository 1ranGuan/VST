import json
import os
import random
import glob
from tqdm import tqdm

# ================= 配置区域 =================

# 输入文件夹路径
INPUT_DIR = os.environ.get("INPUT_DIR", "mhqa/data/sota_data/hld")

# 输出文件路径
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "data/curated_sft_streamvlm_format/hld_1k.jsonl")

# 视频名称到路径的映射文件
NAME2PATH_JSON = os.environ.get("NAME2PATH_JSON", "data_2512/llava_video_name_to_path_index.json")

# System Prompt
SYSTEM_PROMPT = "[System]\nYou are a Streaming Video Analyst.\n"

# ===========================================

def load_name2path(json_path):
    """加载视频路径映射"""
    if not os.path.exists(json_path):
        print(f"Warning: Mapping file not found at {json_path}")
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading mapping file: {e}")
        return {}

def generate_mcq(qa_item):
    """
    生成随机顺序的 MCQ 问题和答案
    """
    question_text = qa_item.get("question", "")
    correct_answer_text = qa_item.get("answer", "")
    distractors = qa_item.get("Distractors", [])
    
    # 1. 构建选项列表
    # 确保有3个干扰项，如果不足可能需要处理，这里假设数据完整
    options_raw = distractors + [correct_answer_text]
    
    # 2. 随机打乱
    random.shuffle(options_raw)
    
    # 3. 分配 A, B, C, D
    labels = ['A', 'B', 'C', 'D']
    formatted_options = []
    final_answer_str = ""
    
    # 截取前4个（防止distractors过多），通常是3错1对
    for i in range(min(len(options_raw), 4)):
        label = labels[i]
        content = options_raw[i]
        
        # 格式化选项字符串: "A. Content"
        option_str = f"{label}. {content}"
        formatted_options.append(option_str)
        
        # 找到正确答案对应的字母
        if content == correct_answer_text:
            final_answer_str = option_str # 例如 "B. Unable to answer"

    # 4. 组合最终的问题字符串
    # 格式要求: Please choose the right answer from the following options: ['A. ...', 'B. ...']
    options_display_str = str(formatted_options) # 转化为 Python List 的字符串表示
    final_question = f"{question_text}\nPlease choose the right answer from the following options: {options_display_str}"
    
    return final_question, final_answer_str

def process_files():
    # 1. 加载映射
    name2path = load_name2path(NAME2PATH_JSON)
    
    # 2. 获取所有 json 文件
    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    print(f"Found {len(json_files)} files in {INPUT_DIR}")
    
    results = []
    
    # 3. 遍历处理
    for file_path in tqdm(json_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            video_name = data.get("video_name", "")
            
            # 获取视频路径，如果映射里没有，暂时保留 name 或者填空
            video_path = name2path.get(video_name, f"MISSING_PATH_FOR_{video_name}")
            
            start_time = data.get("start_time", 0.0)
            end_time = data.get("end_time", 0.0)
            
            # 处理 QA 列表 (通常只有一个，但我们遍历它)
            qa_list = data.get("qa", [])
            if not qa_list:
                continue
                
            for qa_item in qa_list:
                # 生成 MCQ
                q_str, a_str = generate_mcq(qa_item)
                
                # 构造目标数据结构
                entry = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_path,
                                "video_start": start_time,
                                "video_end": end_time
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
                                    [start_time, end_time, q_str, a_str]
                                ]
                            }
                        ]
                    }
                ]
                results.append(entry)
                
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # 4. 写入 JSONL
    print(f"Writing {len(results)} items to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for item in results:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("Done.")

if __name__ == "__main__":
    # 确保输出目录存在（如果是在子目录的话），这里直接写在当前目录
    process_files()
