import random

import re


import json


counting_paraphrases = [
    # --- 最自然/口语化 (Conversational/Natural) ---
    "That makes {TOTAL_COUNT}.",
    "That makes {TOTAL_COUNT} in total.",
    "That's number {TOTAL_COUNT}.",
    "That brings us to {TOTAL_COUNT}.",
    "We're up to {TOTAL_COUNT} now.",
    "And that is {TOTAL_COUNT}.",
    "Counting that one, it's {TOTAL_COUNT}.",
    "That's {TOTAL_COUNT} so far.",
    "We've got {TOTAL_COUNT} now.",
    "That puts us at {TOTAL_COUNT}.",
    "There's number {TOTAL_COUNT}.",
    "Now we have {TOTAL_COUNT}.",
    
    # --- 强调动作/增量 (Action-Oriented) ---
    "This raises the count to {TOTAL_COUNT}.",
    "This pushes the total to {TOTAL_COUNT}.",
    "This brings the total count to {TOTAL_COUNT}.",
    "This increases the total to {TOTAL_COUNT}.",
    "Adding this one brings the count to {TOTAL_COUNT}.",
    "With this addition, the count is {TOTAL_COUNT}.",
    "This bumps the count up to {TOTAL_COUNT}.",
    "This takes the total to {TOTAL_COUNT}.",
    
    # --- 客观/正式/描述性 (Formal/Descriptive) ---
    "The total count is now {TOTAL_COUNT}.",
    "The current count stands at {TOTAL_COUNT}.",
    "The total has reached {TOTAL_COUNT}.",
    "The cumulative count is {TOTAL_COUNT}.",
    "The tally is now {TOTAL_COUNT}.",
    "A total of {TOTAL_COUNT} items have been counted.",
    "The count currently sits at {TOTAL_COUNT}.",
    "{TOTAL_COUNT} items recorded so far.",
    "The aggregate count is {TOTAL_COUNT}.",
    "The running total is {TOTAL_COUNT}.",
    
    # --- 极简/字幕风格 (Concise/Caption Style) ---
    "Total: {TOTAL_COUNT}.",
    "Count: {TOTAL_COUNT}.",
    "Current total: {TOTAL_COUNT}.",
    "Total so far: {TOTAL_COUNT}.",
    "Reaching {TOTAL_COUNT}.",
    "Now at {TOTAL_COUNT}.",
    "Sum: {TOTAL_COUNT}.",
    "{TOTAL_COUNT} total.",
    
    # --- 状态描述 (State-Oriented) ---
    "There are {TOTAL_COUNT} in total now.",
    "We are looking at a total of {TOTAL_COUNT}.",
    "The count has hit {TOTAL_COUNT}.",
    "We have reached a count of {TOTAL_COUNT}.",
    "The number stands at {TOTAL_COUNT}.",
    "Total count equals {TOTAL_COUNT}."
]

def extract_count_info(text):
    # 定义正则表达式
    # 1. r"..." : 原始字符串，防止转义字符冲突
    # 2. (\d+) : 这是核心，\d+ 匹配一个或多个数字，() 表示将其作为一个组提取出来
    # 3. \.    : 匹配句尾的实际句号（因为 . 在正则中代表任意字符，所以需要转义）
    pattern = r"This brings the total count to (\d+) so far\."
    
    # 使用 search 在整个字符串中搜索（match 只能从头匹配）
    match = re.search(pattern, text)
    
    if match:
        # group(0) 是正则匹配到的完整字符串
        full_sentence = match.group(0)
        
        # group(1) 是第一个括号 (...) 内匹配到的内容，即数字字符串
        count_str = match.group(1)
        
        # 转换为 int
        total_count = int(count_str)
        
        return full_sentence, total_count
    else:
        return None, None

def stage2_paraphrase_end(text):
    sentence, total_count = extract_count_info(text)
    if total_count == None:
        return text

    template = random.choice(counting_paraphrases)
    replace_sentence = template.format(TOTAL_COUNT=total_count)
    text = text.replace(sentence, replace_sentence)
    return text

def stage1_filter_f4(text):
    text = text.replace("\n", "")
    return text 

def rewrite_text(text):
    return stage2_paraphrase_end(stage1_filter_f4(text))


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
                print("REPEATED text:", text[i : i + window_size])
                return False
        
        seen_hashes.add(current_hash)
        
    return True


def rewrite(item):
    user_dict, asst_dict = item 
    text_stream = asst_dict["content"][0]["text_stream"]

    new_text_stream = []
    for time_s, tiem_t, text in text_stream:
        new_text_stream.append([time_s, tiem_t, rewrite_text(text)])
    
    asst_dict["content"][0]["text_stream"] = new_text_stream
    return [user_dict, asst_dict]


def read_jsonl(file_path):
    """读取 jsonl 文件，返回 list"""
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 只有非空行才解析
                data_list.append(json.loads(line))
    return data_list

def write_jsonl(file_path, data):
    """将 list 写入 jsonl 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            # ensure_ascii=False 保证中文正常显示
            # end='\n' 或者手动 + '\n' 都可以
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__  == "__main__":
    path = "data/curated_sft_streamvlm_format/counting_cold_start_01.jsonl"
    n_path = "data/curated_sft_streamvlm_format/counting_cold_start_01_rewrite.jsonl"
    data = read_jsonl(path)
    
    new_data = []
    for item in data:
        new_data.append(rewrite(item))
    
    extract_text_func = lambda x : " ".join([c[2] for c in x])

    new_data = [item for item in data if 
        check_repetition_rolling_hash(
            extract_text_func(item[1]["content"][0]["text_stream"]), 80
            )
        ]

    write_jsonl(n_path, new_data)