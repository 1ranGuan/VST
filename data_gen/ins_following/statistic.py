import json
import os 

import statistics
from collections import Counter

def analyze_video_durations(durations):
    if not durations:
        print("没有数据可统计")
        return

    count = len(durations)
    
    # 定义区间计数器
    bins = {
        "< 30s": 0,
        "30s - 60s": 0,
        "60s - 120s": 0,
        "120s - 256s": 0,
        "256s - 512s": 0,
        "> 512s": 0
    }

    # 遍历统计
    for x in durations:
        if x < 30:
            bins["< 30s"] += 1
        elif x < 60:
            bins["30s - 60s"] += 1
        elif x < 120:
            bins["60s - 120s"] += 1
        elif x < 256:
            bins["120s - 256s"] += 1
        elif x < 512:
            bins["256s - 512s"] += 1
        else:
            bins["> 512s"] += 1

    # 打印结果
    print(f"=== 📊 视频时长分布 (Total: {count}) ===")
    print(f"{'区间':<15} | {'数量':<6} | {'占比':<8} | 分布图")
    print("-" * 50)
    
    for label, num in bins.items():
        percent = (num / count) * 100 if count > 0 else 0
        bar_len = int(percent / 2)  # 50% = 25个字符长度
        bar = "█" * bar_len
        print(f"{label:<15} | {num:<6} | {percent:>5.1f}%   | {bar}")

    print("-" * 50)
    print(f"平均时长: {statistics.mean(durations):.2f}s")
    print(f"中位时长: {statistics.median(durations):.2f}s")
    print(f"最短: {min(durations):.2f}s | 最长: {max(durations):.2f}s")



def get_video_durations(file_path):
    duration_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            
            # 使用栈进行深度优先搜索，不管结构多深，找到就算
            stack = [data]
            while stack:
                item = stack.pop()
                
                if isinstance(item, dict):
                    # 核心逻辑：只要字典里有 type: video，就提取信息
                    if item.get("type") == "video":
                        duration = item["video_end"] - item["video_start"]
                        # print(f"视频: {item.get('video')} | 时长: {duration:.2f}s")
                        duration_list.append(duration)
                        break # 假设一行只包含一个主视频，找到就处理下一行
                    
                    # 没找到，就把字典的值加入栈继续找
                    stack.extend(item.values())
                    
                elif isinstance(item, list):
                    # 如果是列表，把里面的元素加入栈继续找
                    stack.extend(item)
    return duration_list

# 运行
TRAIN_DATASET_NAMES=(
    "vm_mcq_5k_train_with_seeks.jsonl",
    "hld_1k_train_with_seeks.jsonl",
    "yes_or_no_3k_train_with_seeks.jsonl",
    "forward_2k_train_with_seeks.jsonl",
    "Streaming_cot_17k_128s_1222_llava_QA_train_with_seeks.jsonl",
    "streamvlm_stage2_17k_0115_train_with_seeks.jsonl",
    "time_chat_online_20k_train_flt_type1_train_with_seeks.jsonl",
    "streamvlm_0114_type2_train_with_seeks.jsonl",
    "counting_cold_start_01_train_with_seeks.jsonl",
    "general_cold_start_02_train_with_seeks.jsonl"
)

LABEL_PATH="data/curated_sft_streamvlm_format"


for dataset_name in TRAIN_DATASET_NAMES:
    print("#"*40)
    print(dataset_name)
    durations = get_video_durations(os.path.join(LABEL_PATH, dataset_name))
        # --- 使用方法 ---
    # 假设这是你上一部提取出来的时长列表
    # durations = [12.35, 24.7, 5.5, 120.1, 35.2, 12.0, 12.8, 9.9, 300.5] 

    analyze_video_durations(durations)

