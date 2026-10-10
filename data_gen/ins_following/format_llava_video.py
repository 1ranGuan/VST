import json
import os
import random
random.seed(333)

def convert_llava_to_stream_format():
    # 配置路径
    input_path = "data_2512/llava_video_open_end.json"
    output_path = "data/curated_sft_streamvlm_format/llava_video_50k.jsonl"

    # 检查并创建输出目录
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print(f"Loading data from {input_path}...")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    data = random.sample(data, 50000)

    print(f"Processing {len(data)} items...")

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item in data:
            # 1. 提取基础信息
            video_path = item.get('video', '')
            duration = item.get('duration', 0.0)
            conversations = item.get('conversations', [])

            # 2. 构建 qa_stream
            qa_stream = []
            
            # 遍历对话，假设是 Human -> GPT 成对出现
            for i in range(0, len(conversations), 2):
                if i + 1 >= len(conversations):
                    break # 跳过不完整的轮次
                
                human_turn = conversations[i]
                gpt_turn = conversations[i+1]

                if human_turn.get('from') == 'human' and gpt_turn.get('from') == 'gpt':
                    # 清洗 <video> 标签
                    question = human_turn.get('value', '').replace('<video>\n', '').replace('<video>', '').strip()
                    answer = gpt_turn.get('value', '')

                    # 注意：原数据没有每轮对话的具体时间戳，这里默认使用全视频时长
                    # 格式: [start, end, Question, Answer]
                    start_time = 0.0 if i==0 else duration
                    qa_stream.append([
                        start_time,
                        duration,
                        question,
                        answer
                    ])

            # 3. 构建目标格式对象
            target_entry = [
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
                            "previous": "[System]\nYou are a Streaming Video Analyst.\n",
                            "title": "",
                            "category": None
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            # 保持和你提供的目标格式一致 (text_steam)
                            "type": "text_steam", 
                            "text_stream": [],
                            "qa_stream": qa_stream
                        }
                    ]
                }
            ]
            # import pdb; pdb.set_trace()

            # 4. 写入 JSONL (一行一个JSON对象)
            f_out.write(json.dumps(target_entry, ensure_ascii=False) + '\n')

    print(f"Conversion complete! Saved to: {output_path}")

if __name__ == "__main__":
    convert_llava_to_stream_format()
