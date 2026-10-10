import json 

def process_json_item(item):
    #user info
    video = item[0]["content"][0]["video"]
    video_start = item[0]["content"][0]["video_start"]
    video_end = item[0]["content"][0]["video_end"] 
    text = item[0]["content"][1]["text"] 
    previous = item[0]["content"][1]["previous"] 
    title = item[0]["content"][1]["title"] 

    #assistant info
    previous_text_stream = [[
        0.,
        1.,
        text
    ]]
    text_steam = previous_text_stream + item[1]["content"][0]["text_stream"] 
    text_steam[1][0] = 1.

    item = {
        "video": video,
        "content": text_steam,
        "title": title
    }
    return item

if __name__ == "__main__":
    input_path = 'streamthink_220qa_1114.jsonl'
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f.readlines(): 
            try:
                json_data = json.loads(line)
                data.append(json_data)
                if not isinstance(json_data, list):
                    raise ValueError("输入JSON必须是数组格式")
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON解析错误: {str(e)}")
    
    new_data = []
    for d in data:
        item = process_json_item(d)
        new_data.append(item) 

    output_path = 'streamthink_eval_1114.jsonl'
    # 处理并写入输出JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for processed in new_data:
            try:
                json.dump(processed, f, ensure_ascii=False)
                f.write('\n')
            except Exception as e:
                print(f"处理条目失败: {str(e)}, 条目内容: {item}")