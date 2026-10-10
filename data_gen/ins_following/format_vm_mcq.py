import json
import os
import re
import random
import ast
random.seed(555)

def convert_clean_jsonl(input_path, output_path):
    # 1. 读取源文件
    with open(input_path, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    
    source_data = [data for data in source_data if data['duration']<256]
    source_data = random.sample(source_data, 500)

    # 2. 写入 JSONL
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item in source_data:
            # --- 提取基础字段 ---
            video_path = item.get('video', '')
            duration = item.get('duration', 0.0)
            
            # --- 提取问题 (Query) ---
            # 优先用 conversations 字段，如果没有则去 'query'  里找
            query = item['conversations'][0]['value'].replace('<video>\n', '').strip()
            if not query:
                query = item.get('query', '')

            # --- 提取答案 (Answer) ---
            answer = ""
            if 'conversations' in item:
                for turn in item['conversations']:
                    if turn['from'] == 'gpt':
                        answer = turn['value']
                        break
            
            try:
                option_str=None
                pattern = r"options:\s*(\[.*?\])"
                match = re.search(pattern, query, re.DOTALL) 
                if match:
                    option_str = match.group(1)
                    query = query.replace(option_str, "")[:-1]
                    # print(option_str)
                    options = ast.literal_eval(option_str)

                    if len(options) == 4:
                        gt_option=answer[0].upper()
                        false_options = random.choice(list(set(["A", "B", "C", "D"]) - set([gt_option])))
                        replace_idx = ord(false_options) - ord('A')
                        options[replace_idx] = f"{false_options}. Unable to answer"

                        query += json.dumps(options)
            except Exception as e:
                print(e)

            # --- 构造目标结构 (无 Prompt) ---
            row_data = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "video_start": 0.0,
                            "video_end": duration
                        },
                        {
                            "type": "text",
                            "text": "",      
                            "previous": "[System]\n"
                "You are a Streaming Video Analyst.\n",    
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
                                [0.0, duration, query, answer]
                            ]
                        }
                    ]
                }
            ]

            # 写入一行
            f_out.write(json.dumps(row_data, ensure_ascii=False) + '\n')

    print(f"转换完成: {output_path}")

if __name__ == "__main__":
    INPUT_FILE = "data_2512/video_marathon_multi_choice.json"
    OUTPUT_FILE = "data/curated_sft_streamvlm_format/vm_mcq_1k_hld.jsonl"
    
    if os.path.exists(INPUT_FILE):
        convert_clean_jsonl(INPUT_FILE, OUTPUT_FILE)
    else:
        print(f"文件不存在: {INPUT_FILE}")
