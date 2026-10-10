import json
import os
import argparse
import tqdm
from typing import List, Dict, Any, Set

# ==========================================
# 基础 I/O 工具函数
# ==========================================

def read_json(file_path: str) -> Any:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def read_jsonl(file_path: str) -> List[Any]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def write_jsonl(data: List[Any], file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# ==========================================
# 核心逻辑函数
# ==========================================

def build_thinking_index(thinking_data: List[Any]) -> Dict[str, Any]:
    """
    构建 Thinking 数据集的索引。
    逻辑：对于同一个视频 ID，保留 video_start 最大的那个 item。
    返回: { video_id: {'start': float, 'end': float, 'item': dict} }
    """
    print('Constructing thinking index map...')
    thinking_map = {}
    
    for item in tqdm.tqdm(thinking_data, desc="Indexing Thinking Data"):
        # item 结构通常为 [{'role': 'user', ...}, {'role': 'assistant', ...}]
        user_msg = item[0]
        
        # 提取视频信息
        video_info = None
        for content in user_msg.get('content', []):
            if content.get('type') == 'video':
                video_info = content
                break
        
        if not video_info:
            continue

        video_path = video_info.get('video', '')
        video_start = video_info.get('video_start', 0.0)
        video_end = video_info.get('video_end', 0.0)
        
        # ID 处理逻辑 (此处保持与原代码一致，使用完整路径作为 ID)
        # 如果需要特殊处理文件名，可在此处修改
        vid_id = video_path 

        # 寻找该视频 ID 下，video_start 最大的那个 item
        if vid_id not in thinking_map:
            thinking_map[vid_id] = {
                'start': video_start,
                'end': video_end,
                'item': item
            }
        else:
            if video_start > thinking_map[vid_id]['start']:
                thinking_map[vid_id] = {
                    'start': video_start,
                    'end': video_end,
                    'item': item
                }
    
    print(f'Thinking index built. Unique videos: {len(thinking_map)}')
    return thinking_map

def process_qa_data(qa_data: List[Any], thinking_map: Dict[str, Any]) -> List[Any]:
    """
    将 QA 数据合并到 Thinking 数据结构中。
    """
    print('Filtering and merging QA stream...')
    curated_data = []
    sampled_ids: Set[str] = set()

    for item_qa in tqdm.tqdm(qa_data, desc="Processing QA"):
        qa_id = item_qa.get('video')
        
        # 查重
        if qa_id in sampled_ids:
            continue
        
        # 匹配 Thinking 数据
        if qa_id in thinking_map:
            sampled_ids.add(qa_id)
            
            target_entry = thinking_map[qa_id]
            item_think = target_entry['item'] # 这是一个引用
            
            # 关键修改：修改类型并初始化 qa_stream
            # 注意：这里直接修改了 thinking_map 中的引用对象
            # 如果同一个 thinking item 只需要被匹配一次，这是没问题的
            
            # 1. 修改类型
            item_think[1]['content'][0]['type'] = 'qa_stream'
            
            # 2. 初始化 qa_stream 列表 (防止 KeyError)
            item_think[1]['content'][0]['qa_stream'] = []
            
            # 3. 获取时间戳 (原代码逻辑：使用 thinking 片段的 video_end)
            video_end = item_think[0]['content'][0]['video_end']

            # 4. 处理对话
            conversations = item_qa.get('conversations', [])
            
            # 遍历对话，步长为2 (User, GPT)
            for i in range(0, len(conversations), 2):
                if i + 1 < len(conversations):
                    human_text = conversations[i]['value'].replace('<video>\n', '')
                    gpt_text = conversations[i+1]['value']
                    
                    # 格式: [start, end, Q, A]
                    # 原代码逻辑中，start 和 end 都使用了 video_end
                    item_think[1]['content'][0]['qa_stream'].append(
                        [video_end, video_end, human_text, gpt_text]
                    )
            
            curated_data.append(item_think)

    return curated_data

# ==========================================
# 主程序
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Merge QA datasets with Thinking dataset based on Video ID.")
    
    # 输入参数
    parser.add_argument('--qa_files', nargs='+', required=True, help='Paths to QA JSON files (supports multiple).')
    parser.add_argument('--thinking_file', type=str, required=True, help='Path to the Thinking JSONL file.')
    
    # 输出参数
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the result JSONL file.')
    
    args = parser.parse_args()

    # 1. 加载并合并所有 QA 数据
    all_qa_data = []
    for qa_path in args.qa_files:
        print(f"Loading QA file: {qa_path}")
        data = read_json(qa_path)
        all_qa_data.extend(data)
    print(f"Total QA records loaded: {len(all_qa_data)}")

    # 2. 加载 Thinking 数据
    print(f"Loading Thinking file: {args.thinking_file}")
    thinking_data = read_jsonl(args.thinking_file)
    
    # 3. 构建索引
    thinking_map = build_thinking_index(thinking_data)

    # 4. 处理数据
    final_data = process_qa_data(all_qa_data, thinking_map)
    print(f"Final dataset size: {len(final_data)}")

    # 5. 写入结果
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    write_jsonl(final_data, args.output_file)
    print(f"Saved to: {args.output_file}")

if __name__ == '__main__':
    main()
