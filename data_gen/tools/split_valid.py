import json 
import os

def filter_func(item):
    '''
        如果满足要求，return True
        如果有问题，return False
    '''
    #视频起止时间超过70s
    def overlong(item):
        if item[0]['content'][0]['video_end'] - item[0]['content'][0]['video_start'] >= 900:
            return False 

    if overlong(item) == False:
        return False 
    
    return True

def filter_data(json_data):
    res_data = []
    for item in json_data:
        if filter_func(item):
            res_data.append(item)
    return res_data 

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {str(e)}")
    return data

def write_jsonl(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")

def get_video_id(video):
    # 判断是否为字符串
    if isinstance(video, str):
        return video
    # 判断是否为列表
    elif isinstance(video, list):
        if len(video) > 0:
            # 获取第一个文件路径所在的目录
            return os.path.dirname(video[0])
        else:
            return ""
    return ""

def sample_valid_train(data, val_ratio=0.05):
    video_question_map = {}
    for i, item in enumerate(data):
        video = get_video_id(item[0]['content'][0]['video'])
        if video in video_question_map:
            video_question_map[video].append(i)
        else:
            video_question_map[video] = [i]
    print('total videos: ', len(video_question_map))
    total_video_num = len(video_question_map)

    print('sampling videos for validation... ')
    valid_videos_names = list(video_question_map.keys())[:int(val_ratio*total_video_num)]
    print('sampled videos for validation: ', int(val_ratio*total_video_num))
    valid_question_ids = []
    for video in valid_videos_names:
        valid_question_ids += video_question_map[video]
    valid_data = []
    for id in valid_question_ids:
        valid_data.append(data[id])
    print('sampled qas for validation: ', len(valid_data))


    print('sampling videos for training... ')
    train_videos_names = list(video_question_map.keys())[int(val_ratio*total_video_num):]
    print('sampled videos for training: ', len(train_videos_names))
    train_question_ids = []
    for video in train_videos_names:
        train_question_ids += video_question_map[video]
    train_data = []
    for id in train_question_ids:
        train_data.append(data[id])
    print('sampled qas for training: ', len(train_data))
    return train_data, valid_data
    

def sample_to_ids(json_data, ids):
    return [json_data[id] for id in ids]

from collections import Counter


import os
import argparse
from collections import Counter
from typing import List, Tuple, Any

# ==========================================
# 核心逻辑
# ==========================================

def analyze_data(json_data: List[Any]):
    """
    统计并打印数据的分布情况 (Debug模式用)
    """
    count_num = []
    duration_count = []
    
    print("Running debug analysis...")
    try:
        for item in json_data:
            # 假设 item 结构为 (video_info, text_info, ...) 
            # 根据原代码逻辑保留索引访问
            text_len = len(item[1]['content'][0]['text_stream'])
            duration = int(item[0]['content'][0]['video_end'] - item[0]['content'][0]['video_start'])
            
            duration_count.append(duration)
            count_num.append(text_len)
            
        number_counts = Counter(count_num)
        duration_counts = Counter(duration_count)
        
        print(f"Text stream length counts: {number_counts}")
        print(f"Duration counts: {duration_counts}")
        print(f'Number of data after invalid ids check: {len(json_data)}')
        
    except (IndexError, KeyError, TypeError) as e:
        print(f"Error during analysis: {e}. Please check data structure.")

def main():
    parser = argparse.ArgumentParser(description="Process video streaming data for training and validation.")
    
    # 命令行参数定义
    parser.add_argument(
        '--input_path', 
        type=str, 
        required=True,
        help='Path to the input JSONL file.'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=None, 
        help='Directory to save output files. Defaults to input file directory.'
    )
    parser.add_argument(
        '--val_ratio', 
        type=float, 
        default=0.03, 
        help='Ratio of data to use for validation (default: 0.03).'
    )
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help='Enable debug mode to print data statistics.'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        default='with_seeks',
        help='Suffix to append to output filenames (default: with_seeks).'
    )

    args = parser.parse_args()

    # 1. 路径检查
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    # 确定输出目录
    if args.output_dir is None:
        output_dir = os.path.dirname(args.input_path)
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

    # 2. 读取数据
    json_data = read_jsonl(args.input_path)
    print(f'Total data loaded: {len(json_data)}')

    # 3. Debug 分析
    if args.debug:
        analyze_data(json_data)

    # 4. 数据过滤
    json_data = filter_data(json_data)
    print(f'Data count after filtering: {len(json_data)}')

    # 5. 划分训练集和验证集
    train_data, valid_data = sample_valid_train(json_data, val_ratio=args.val_ratio)
    print(f"Split result - Train: {len(train_data)}, Valid: {len(valid_data)}")

    # 6. 生成输出文件名
    # 获取输入文件的文件名（不带扩展名），例如 "Streaming_cot_vm_60k_1226"
    base_name = os.path.splitext(os.path.basename(args.input_path))[0]
    
    train_filename = f"{base_name}_train_{args.suffix}.jsonl"
    valid_filename = f"{base_name}_valid_{args.suffix}.jsonl"
    
    train_path = os.path.join(output_dir, train_filename)
    valid_path = os.path.join(output_dir, valid_filename)

    # 7. 写入文件
    write_jsonl(train_data, train_path)
    write_jsonl(valid_data, valid_path)
    
    print(f"Successfully saved to:\n  - {train_path}\n  - {valid_path}")

if __name__ == "__main__":
    main()

