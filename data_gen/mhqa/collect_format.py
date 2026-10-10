import json
import os
import re
from typing import List, Dict
from tqdm import tqdm

# ================= 配置区域 =================
# 源数据路径
CAPTION_ROOT = os.environ.get("CAPTION_ROOT", "mhqa/data/captions")            # stage_3_add_discription.py 输出
QA_ROOT = os.environ.get("QA_ROOT", "mhqa/data/qa_dataset_v0120_step1")       # stage_1_generate_qa.py 输出

# 输出文件路径
write_qa = True

repetition_count = 0

OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "mhqa/data/streamvlm_stage2_75k_vm_0124.jsonl")

NAME2PATH_JSON = os.environ.get("NAME2PATH_JSON", "data_2512/video_name_to_path_index.json")  # tools/build_name2path.py 输出
with open(NAME2PATH_JSON, "r", encoding="utf-8") as f:
    name2path = json.load(f)

# 滑动窗口最大时间 (秒)
def MAX_WINDOW_T(duration):
    '''
        短视频:1-17min, 以128s为chunk
        中视频:17-34min, 以256s为chunk
        长视频:>34min, 以duration/8为chunk
    '''
    if duration <= 128*8:
        return 128
    elif duration <= 256*8:
        print(duration//8)
        return duration//8
    
def check_repetition_rolling_hash(text, window_size=10):
    if len(text) < window_size:
        return False
        
    seen_hashes = set()
    
    # Python 内置的 hash() 对于字符串已经非常优化
    # 虽然不是严格的滚动更新，但在 Python 中通常比手写 Rabin-Karp 更快
    for i in range(len(text) - window_size + 1):
        # 这里的 hash() 计算是 O(K)
        current_hash = hash(text[i : i + window_size])
        
        if current_hash in seen_hashes:
            # 为了防止哈希碰撞（极小概率），这里可以二次确认
            # 实际工程中，对于去重任务，通常可以直接信任 hash
            if text[i : i + window_size] in text[:i]: # Double check

                print("#"*100)
                print(text)
                print(text[i : i + window_size])
                return True
        
        seen_hashes.add(current_hash)
        
    return False


# 严格按照您的要求初始化 System Prompt
SYSTEM_PROMPT_INIT = "[System]\nYou are a Streaming Video Analyst.\n"

# ===========================================

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON: {path}")
        return None

def extract_part_index(filename):
    """从 part_0.mp4 中提取数字 0"""
    match = re.search(r'part_(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1

def process_captions(caption_data: List[Dict]) -> List[List]:
    """
    提取并去重Caption数据。
    返回格式: [[start, end, caption], ...] 按时间排序
    """
    caption_data = caption_data["captions"]
    unique_captions = []
    for item in caption_data:
        start = float(item.get("start_time", 0.0))
        end = float(item.get("end_time", 0.0))
        text = item.get("caption", "").strip()
        unique_captions.append([start, end, text])
    
    # 按开始时间排序
    # unique_captions.sort(key=lambda x: x[0])
    
    return unique_captions

def get_qa_data(video_name):
    """读取QA数据"""
    path = os.path.join(QA_ROOT, f"{video_name}.json")
    data = load_json(path)
    video_path = None
    if not data:
        return video_path, []
    
    # data 是 list[dict], 提取里面的 qa 部分
    qa_list = []
    for item in data:
        if video_path==None:
            video_path = item["video_name"]
        if "qa" in item:
            qa_list.append(item["qa"])
            
    return video_path, qa_list

def save_to_jsonl(data, output_file):
    """
    将数据保存为 JSONL 格式
    :param data: 包含所有数据的列表
    :param output_file: 输出文件路径
    """
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data:
            # ensure_ascii=False 保证中文等字符正常显示
            # 每条数据占一行
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def main():
    global repetition_count

    all_formatted_data = []
    
    if not os.path.exists(CAPTION_ROOT):
        print(f"Directory not found: {CAPTION_ROOT}")
        return

    video_names = [d.replace('.json', '') for d in os.listdir(CAPTION_ROOT)]
    print(f"Found {len(video_names)} videos. Processing...")

    for video_name in tqdm(video_names):

        # 1. 读取Caption
        caption_file_path = os.path.join(CAPTION_ROOT, f"{video_name}.json")
        caption_source = load_json(caption_file_path)
        
        if not caption_source:
            continue

        caption_stream = process_captions(caption_source)
        if not caption_stream:
            continue
        
        text = "".join([ele[2] for ele in caption_stream])
        if check_repetition_rolling_hash(text, 80):
            #删除caption_file_path
            os.remove(caption_file_path)
            repetition_count += 1
            continue

        video_duration = caption_stream[-1][1] - caption_stream[0][0]
        max_window_t = MAX_WINDOW_T(video_duration)
        if max_window_t>256:
            print('max_window_t: ', max_window_t)

        # 2. 读取QA
        video_path, qa_source = get_qa_data(video_name)

        video_path = name2path.get(video_name, video_path)


        # 3. 滑动窗口切分
        chunks = []
        current_chunk = []
        if caption_stream:
            chunk_start_time = caption_stream[0][0]
        
        for cap in caption_stream:
            start, end, text = cap
            
            if not current_chunk:
                current_chunk.append(cap)
                chunk_start_time = start
            else:
                # 检查窗口大小
                if end - chunk_start_time <= max_window_t:
                    current_chunk.append(cap)
                else:
                    chunks.append(current_chunk)
                    current_chunk = [cap]
                    chunk_start_time = start
        
        if current_chunk:
            chunks.append(current_chunk)

        # 4. 生成目标格式
        # 严格按照要求初始化 previous
        previous_history = SYSTEM_PROMPT_INIT
        
        for idx, chunk in enumerate(chunks):
            is_last_chunk = (idx == len(chunks) - 1)
            
            chunk_start = chunk[0][0]
            chunk_end = chunk[-1][1]
            
            # 构建 qa_stream (仅在最后一段填充)
            qa_stream_formatted = []
            if write_qa:
                if is_last_chunk and qa_source:
                    for qa_item in qa_source:
                        q = qa_item.get("question", "")
                        a = qa_item.get("answer", "")
                        qa_stream_formatted.append([chunk_end, chunk_end, q, a])
            
            # 构造 user content
            # 注意：previous 使用的是当前 chunk *之前* 累积的内容
            user_content = [
                {
                    "type": "video",
                    "video": video_path,
                    "video_start": chunk_start,
                    "video_end": chunk_end
                },
                {
                    "type": "text",
                    "text": "",
                    "previous": previous_history, # 这里直接使用累积的字符串
                    "title": "",
                    "category": None
                }
            ]
            
            # 构造 assistant content
            assistant_content = [
                {
                    "type": "text_steam",
                    "text_stream": chunk,
                    "qa_stream": qa_stream_formatted
                }
            ]
            
            # 添加到总列表
            all_formatted_data.append([
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ])

            # 5. 更新 previous (为下一个 chunk 准备)
            # 按照指令：previous += f"Time={start_time:.1f}-{end_time:.1f}s {caption}\n"
            for cap in chunk:
                c_start, c_end, c_text = cap
                previous_history += f"Time={c_start:.1f}-{c_end:.1f}s {c_text}\n"

    print(repetition_count)
    # 保存结果
    save_to_jsonl(all_formatted_data, OUTPUT_FILE)
    
    print(f"Processing complete. Saved {len(all_formatted_data)} samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    repetition_count = 0
    main()
    print(repetition_count)

