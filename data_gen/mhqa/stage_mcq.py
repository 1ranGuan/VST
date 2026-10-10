import requests
import json
import re
import time
import os
import glob
import random
import argparse
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from tqdm import tqdm
import traceback
from typing import List, Dict, Any, Tuple

# ================= 配置区域 =================

API_URL = os.environ.get("VST_LLM_API_URL", "")
API_KEY = os.environ.get("VST_LLM_API_KEY", "")
MODEL_NAME = "gemini-3-flash"
MAX_WORKERS = 16                             # 并行进程数
random.seed(333)

# ================= 类定义 =================

class VideoStreamReasoningGenerator:
    def __init__(self, api_url, api_key, model="gemini-3-flash", video_name=None, output_dir=None):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key
        }
        self.video_name = video_name
        self.output_dir = output_dir

    def _call_api(self, prompt, system_prompt="You are a helpful AI assistant."):
        payload = {
            "model": self.model,
            "key_id": 1,
            "contents": prompt,
            "system_prompt": system_prompt
        }

        try:
            # 增加超时时间以防生成长文本
            response = requests.post(self.api_url, headers=self.headers, data=json.dumps(payload), timeout=120) 
            if response.status_code == 200:
                resp_json = response.json()
                
                if "results" in resp_json and isinstance(resp_json["results"], list) and len(resp_json["results"]) > 0:
                    content = resp_json["results"][0]
                    return self._clean_json_string(content)
                
                print(f"Warning: Unexpected JSON structure: {resp_json.keys()}")
                return None
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            print(f"Request Failed: {e}")
            return None

    def _clean_json_string(self, text):
        if not text: return None
        text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
        text = text.strip()
        return text

    def generate_distractors(self, question, answer, cot):
        """
        调用 API 生成 3 个干扰项
        """
        system_prompt = "You are an expert at creating multiple-choice questions for video understanding tasks."
        prompt = f"""
I have a question and a correct answer based on a video. 
Please generate 3 plausible but INCORRECT answers (distractors).
The distractors can be factually wrong, irrelevant, or partially correct but essentially wrong details compared to the correct answer.
They should be similar in length and style to the correct answer.

Question: {question}
Correct Answer: {answer}
Chain of Thought (Context): {cot}

Output ONLY a raw JSON list of strings containing exactly 3 distractors. 
Example format: ["Wrong answer 1", "Wrong answer 2", "Wrong answer 3"]
"""
        response_text = self._call_api(prompt, system_prompt)
        
        if response_text:
            try:
                distractors = json.loads(response_text)
                if isinstance(distractors, list) and len(distractors) >= 3:
                    return distractors[:3] # 确保只取前3个
            except json.JSONDecodeError:
                print(f"JSON Decode Error: {response_text}")
        return None

    def process_item(self, item):
        """
        处理单个数据条目：生成选项，重组结构
        """
        try:
            qa_data = item.get("qa", {})
            original_question = qa_data.get("question", "")
            original_answer = qa_data.get("answer", "")
            cot = qa_data.get("cot", "")

            # 1. 生成干扰项
            distractors = self.generate_distractors(original_question, original_answer, cot)
            if not distractors:
                return None # 生成失败则跳过

            # 2. 构建选项列表并打乱
            options_list = [original_answer] + distractors
            random.shuffle(options_list)

            # 3. 确定正确答案的字母
            letters = ['A', 'B', 'C', 'D']
            correct_index = options_list.index(original_answer)
            correct_letter = letters[correct_index]

            # 4. 格式化选项字符串
            # 格式例如: (A) The orange bell pepper...
            formatted_options = [f"({letters[i]}) {opt}" for i, opt in enumerate(options_list)]
            options_str = "\n".join(formatted_options)

            # 5. 构建新的 Question 字符串
            option_prompt = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option."
            post_prompt = "Answer with the option's letter from the given choices directly."
            
            # 最终拼接逻辑: option_prompt + question + post_prompt + options
            final_question = f"{option_prompt}\n{original_question}\n{post_prompt}\n{options_str}"

            # 6. 组装新数据
            new_item = item.copy()
            new_item["qa"] = {
                "question": final_question,
                "cot": cot,
                "answer": correct_letter,
                "reasoning_type": qa_data.get("reasoning_type", ""),
                "original_data": {
                    "question": original_question,
                    "options": options_list, # 记录包含正确答案和干扰项的列表
                    "answer": original_answer
                }
            }
            return new_item

        except Exception as e:
            print(f"Error processing item: {e}")
            return None

# ================= 流程处理函数 =================

def process_file_wrapper(file_path, output_dir):
    """
    单个文件处理函数，用于多进程调用
    """
    try:
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)

        # 实例化生成器
        generator = VideoStreamReasoningGenerator(
            api_url=API_URL, 
            api_key=API_KEY, 
            model=MODEL_NAME,
            video_name=filename
        )

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        new_data = []
        for item in data:
            processed_item = generator.process_item(item)
            if processed_item:
                new_data.append(processed_item)
            else:
                print(f"Skipping item in {filename} due to API failure.")

        # 只有当有数据处理成功时才写入
        if new_data:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=4, ensure_ascii=False)
            return True
        return False

    except Exception as e:
        print(f"Critical error processing file {file_path}: {e}")
        traceback.print_exc()
        return False

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Open-Ended QA to Multiple Choice via API")
    
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default='qa_dataset/',
        help='Input directory containing source JSON files'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='qa_dataset_filtered/',
        help='Output directory for processed JSON files'
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # 获取所有 JSON 文件
    json_files = glob.glob(os.path.join(args.input_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files to process.")

    # 多进程处理
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for json_file in json_files:
            futures.append(
                executor.submit(process_file_wrapper, json_file, args.output_dir)
            )

        # 进度条监控
        success_count = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            if future.result():
                success_count += 1

    print(f"Processing complete. Successfully processed {success_count}/{len(json_files)} files.")

if __name__ == "__main__":
    main()
