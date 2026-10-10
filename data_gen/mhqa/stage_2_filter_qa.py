import requests
import json
import re
import time
import os
import glob
import argparse
import random 
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from tqdm import tqdm

# ================= 配置区域 =================
# INPUT_DIR = 'qa_dataset/'
# OUTPUT_DIR = 'qa_dataset_filtered/'
API_URL = os.environ.get("VST_LLM_API_URL", "")
API_KEY = os.environ.get("VST_LLM_API_KEY", "")
MODEL_NAME = "gemini-3-flash"
MAX_WORKERS = 15                             # 并行进程数，根据你的 CPU 核数调整
random.seed(333)


class qaFilter:
    def __init__(self, api_url, api_key, model="gpt-4o"):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key
        }

    def _call_api(self, prompt, system_prompt="You are a helpful AI assistant."):
        payload = {
            "model": self.model,
            "key_id": 1,
            "contents": prompt,
            "system_prompt": system_prompt
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, data=json.dumps(payload), timeout=60)
            if response.status_code == 200:
                resp_json = response.json()
                
                # === 针对你提供的 JSON 格式进行解析 ===
                # 格式: {"code": 200, "results": ["生成的文本内容..."], "usage": ...}
                if "results" in resp_json and isinstance(resp_json["results"], list) and len(resp_json["results"]) > 0:
                    content = resp_json["results"][0]  # 提取列表里的第一个字符串
                    return self._clean_json_string(content)
                
                # 如果找不到 results 字段，打印调试信息
                print(f"Warning: Unexpected JSON structure: {resp_json.keys()}")
                return None
            
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            print(f"Request Failed: {e}")
            return None

    def _clean_json_string(self, text):
        """
        清理模型返回的字符串，提取JSON部分
        """
        if not text: return None
        # 移除 markdown 代码块标记
        text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
        text = text.strip()
        return text

    # ================= Stage 1: 本地文本检查 =================
    def _stage_1_text_check(self, item):
        """
        检查是否包含乱码或非英文字符
        """
        def is_clean(s):
            if not s: return False
            # 检查是否包含 CJK 字符 (中日韩)
            if re.search(r'[\u4e00-\u9fff]', s):
                return False
            return True

        # 1. Check path_nodes
        for node in item.get('path_nodes', []):
            if not is_clean(node):
                return False, "Path nodes contain non-English chars"

        # 2. Check QA
        qa = item.get('qa', {})
        if not is_clean(qa.get('question')) or \
           not is_clean(qa.get('cot')) or \
           not is_clean(qa.get('answer')):
            return False, "QA content contains non-English chars"

        return True, "Pass"

    # ================= Stage 2: 世界知识依赖检查 (API) =================
    def _stage_2_world_knowledge(self, question, random_clip_text, answer):
        """
        Stage 2 改良版：双步验证
        1. 尝试回答 (Generate)
        2. 评分 (Evaluate)
        """
        
        # ================= Step 1: 尝试回答 =================
        # 目的：看模型仅凭常识能回答到什么程度
        gen_prompt = f"""
            You are an AI assistant. 
            Question: "{question}"
            Context: "{random_clip_text}"

            Task: Answer this question using ONLY your general world knowledge and common sense. 
            - Do NOT assume you have access to any video.
            - If the question asks for specific visual details (e.g., "what color is the cat"), you should admit you don't know or make a best guess based on the text context only.
            - Give your answer directly and Keep the answer concise.
            """
        generated_answer = self._call_api(gen_prompt, system_prompt="You are a helpful assistant.")
        
        if not generated_answer:
            # 如果第一步就失败（如API报错），为了安全起见，可以选择报错或放行
            # 这里选择报错，避免脏数据漏过
            return False, "API Error Stage 2 (Generation)"

        # ================= Step 2: 评分 =================
        # 目的：评估刚才的回答是否过于完美（意味着问题太简单，不需要视频）
        score_prompt = f"""
            You are an impartial judge evaluating the semantic equivalence of two answers.

            Input Data:
            [Question]: {question}
            [Ground Truth Answer]: {answer}
            [Candidate Answer]: {generated_answer}

            Task:
            Compare the [Candidate Answer] against the [Ground Truth Answer] in the context of the [Question].
            Rate the accuracy and completeness of the Candidate Answer relative to the Ground Truth on a scale of 1 to 5.

            Scoring Criteria:
            1: **Incorrect / Information Missing**. The Candidate Answer is completely wrong, irrelevant, or explicitly states "I don't know".
            2: **Low Accuracy**. The Candidate Answer contains very little correct information or is mostly hallucinated/wrong compared to the Ground Truth.
            3: **Partially Correct**. The Candidate Answer captures the general topic or some keywords but misses the specific core details present in the Ground Truth.
            4: **Mostly Correct**. The Candidate Answer aligns with the Ground Truth in meaning but may lack minor details or precision.
            5: **Correct / Equivalent**. The Candidate Answer conveys the same information and meaning as the Ground Truth.

            Output strictly in JSON format:
            {{
                "reasoning": "Brief objective comparison of the two answers",
                "score": int
            }}
            """
        response_str = self._call_api(score_prompt, system_prompt="You are a strict evaluator. Output JSON only.")

        
        if not response_str:
            return False, "API Error Stage 2 (Scoring)"

        try:
            result = json.loads(response_str)
            score = result.get('score', 0)
            reasoning = result.get('reasoning', 'No reasoning provided')
            if score > 3:
                # 记录一下生成的答案片段，方便后续分析为何被过滤
                short_ans = generated_answer[:50].replace('\n', ' ') + "..."
                return False, f"World Knowledge Score too high ({score}/5). Ans: {short_ans}"
            
            return True, f"Pass (Score: {score}/5)"
            
        except json.JSONDecodeError:
            return False, "JSON Parse Error Stage 2"


    # ================= Stage 3: CoT 逻辑与片段还原 (API) =================
    def _stage_3_logic_and_extraction(self, item):
        """
        Stage 3 增强版：逻辑自洽 + 实体强关联检查 + 片段还原
        """
        question = item['qa']['question']
        cot = item['qa']['cot']
        answer = item['qa']['answer']
        
        # 构建包含实体信息的片段列表
        clips_info = []
        for idx, entry in enumerate(item['path_entries']):
            clips_info.append({
                "id": idx,
                "time": f"{entry['start_time']:.1f}-{entry['end_time']:.1f}s",
                "triplet": f"({entry['head']}) --[{entry['rel']}]--> ({entry['tail']})", # 强调三元组结构
                "desc": entry['desc']
            })

        prompt = f"""
            You are a strict Graph-Grounding Verifier for a Video QA dataset.

            Input Data:
            [Question]: {question}
            [CoT]: {cot}
            [Answer]: {answer}
            [Visual Knowledge Path (Clips)]: {json.dumps(clips_info)}

            Your Goal: Verify if the QA reasoning is TIGHTLY GROUNDED in the provided Visual Knowledge Path entities.

            Tasks:
            1. **Entity Grounding Check (Crucial)**: 
            - Do the Question, CoT, and Answer explicitly discuss the specific entities (Head/Tail) and actions (Rel) listed in the [Visual Knowledge Path]?
            - **FAIL** if the QA talks about generic concepts (e.g., "objects", "things", "process") without naming the specific entities (e.g., "flour", "meat", "mix").
            - **FAIL** if the QA discusses entities NOT present in the path (Hallucination).

            2. **Logic Consistency Check**: 
            - Is the CoT logically consistent with the Answer?
            - Does the CoT accurately reflect the sequence of events described in the clips' triplets?

            3. **Clip Extraction**: 
            - Identify exactly which Clip IDs are necessary to support the CoT.
            - STRICTLY match the events in CoT to the 'triplet' and 'desc' in clips.

            Output strictly JSON:
            {{
                "grounding_score": int, // 1-5. 1=Irrelevant/Hallucinated, 3=Vague/Generic, 5=Strongly aligned with specific entities.
                "logic_valid": boolean, // True only if reasoning is sound.
                "reasoning": "Short explanation of the score",
                "used_clip_ids": [list of integers]
            }}
            """
        response_str = self._call_api(prompt, system_prompt="You are a strict data auditor. Output JSON only.")
        if not response_str:
            return False, "API Error Stage 3", []

        try:
            result = json.loads(response_str)
            
            # 1. 实体强关联检查 (Entity Grounding)
            # 只有分数 >= 4 才算通过（即要求 QA 必须紧密围绕实体展开）
            grounding_score = result.get('grounding_score', 0)
            if grounding_score <= 1:
                return False, f"Weak Entity Grounding (Score: {grounding_score}/5). Reason: {result.get('reasoning')}", []

            # 2. 逻辑检查
            if not result.get('logic_valid', False):
                return False, "CoT Logic Invalid", []

            # 3. 片段数量检查
            used_ids = result.get('used_clip_ids', [])
            unique_ids = sorted(list(set(used_ids)))
            
            if len(unique_ids) <= 1:
                return False, f"Not enough clips used (Count: {len(unique_ids)} <= 2)", unique_ids

            # --- 通过检查，更新 item ---
            # 仅保留被引用的 path_entries，清洗数据
            # 注意：这里会修改原数据结构，确保你确实想删除未引用的片段
            item['path_entries'] = [entry for idx, entry in enumerate(item['path_entries']) if idx in unique_ids]
            
            # 可选：把验证过的 ID 存入 meta
            item['meta_info'] = item.get('meta_info', {})
            item['meta_info']['grounding_score'] = grounding_score
            
            return True, "Pass", item

        except json.JSONDecodeError:
            return False, "JSON Parse Error Stage 3", []


        except json.JSONDecodeError:
            return False, "JSON Parse Error Stage 3", []
    

    # ================= 主流程 =================
    def filter_single_item(self, item):
        """
        流水线处理单个 Item
        """
        # --- Stage 1: Text Check ---
        s1_pass, s1_reason = self._stage_1_text_check(item)
        if not s1_pass:
            return False, f"Stage 1 Fail: {s1_reason}", None

        # --- Stage 2: World Knowledge Check ---
        # 仅传入 Question和随机一个clip的caption
        random_caption = [item["video_clip_caption"] for item in random.sample(item["path_entries"], 1)]
        random_caption = '\n'.join(random_caption)
        s2_pass, s2_reason = self._stage_2_world_knowledge(item['qa']['question'], random_caption, item['qa']['answer'])
        if not s2_pass:
            return False, f"Stage 2 Fail: {s2_reason}", None

        # --- Stage 3: Logic & Extraction ---
        s3_pass, s3_reason, item = self._stage_3_logic_and_extraction(item)
        if not s3_pass:
            return False, f"Stage 3 Fail: {s3_reason}", None

        return True, "All Passed", item

def process_batch(data_list):
    """
    批处理
    """
    results = []
    print(f"Processing {len(data_list)} items...")
    
    for i, item in enumerate(data_list):
        print(f"--- Item {i} ---")
        pipeline = qaFilter(API_URL, API_KEY, model=model)
        is_valid, reason, processed_item = pipeline.filter_single_item(item)
        
        if is_valid:
            print(f"✅ KEEP. Reason: {reason}")
            results.append(processed_item)
        else:
            print(f"❌ DROP. Reason: {reason}")
    
    return results

# # ================= 使用示例 =================
# if __name__ == "__main__":
#     # 你的 API 配置
    
#     # model = "gemini-2.0-flash"
    
#     # 实例化
    
    
#     # 模拟数据
#     p = 'qa_dataset/ZjMvwTpzQF0_4fps_480p.json'
#     data = json.load(open(p, 'r'))

#     # 运行
#     # 注意：如果没有真实的 API Server，这里会报错或需要 mock
#     filtered_list = process_batch(data)


def log_listener(queue):
    """
    专门负责打印日志的线程。
    它从队列中获取消息，并使用 tqdm.write 打印，这样不会破坏进度条。
    """
    while True:
        record = queue.get()
        if record is None:  # 结束信号
            break
        tqdm.write(record)  # 关键：使用 tqdm.write 代替 print

# ================= Worker 函数 =================
def process_single_json_file(file_path, log_queue):
    """
    子进程执行函数
    """
    file_name = os.path.basename(file_path)
    
    # 定义一个本地的 log 函数，简化调用
    def log(msg):
        log_queue.put(f"[{file_name}] {msg}")

    # log(f"Start processing...") # 可选：打印开始日志

    try:
        # 1. 初始化过滤器
        pipeline = qaFilter(API_URL, API_KEY, model=MODEL_NAME)
        
        # 2. 读取数据
        with open(file_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
            
        if not isinstance(data_list, list):
            log("Skipped: Not a list")
            return file_name, 0, 0, "Skipped"

        valid_items = []
        dropped_count = 0
        
        # 3. 遍历处理
        for i, item in enumerate(data_list):
            try:
                is_valid, reason, processed_item = pipeline.filter_single_item(item)
                
                if is_valid:
                    # 实时打印保留信息
                    log(f"Item {i} ✅ KEEP | {reason}")
                    valid_items.append(processed_item)
                else:
                    # 实时打印过滤信息
                    log(f"Item {i} ❌ DROP | {reason}")
                    dropped_count += 1
                    
            except Exception as e:
                log(f"Item {i} ⚠️ ERROR | {str(e)}")
                dropped_count += 1

        # 4. 保存结果
        if valid_items:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            out_path = os.path.join(OUTPUT_DIR, file_name)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(valid_items, f, indent=2, ensure_ascii=False)
            log(f"Saved {len(valid_items)} items to {out_path}")
        else:
            log(f"No valid items left, skipping save.")

        return file_name, len(data_list), len(valid_items), "Success"

    except Exception as e:
        err_msg = f"File Error: {str(e)}"
        log(err_msg)
        return file_name, 0, 0, err_msg

def parse_args():
    parser = argparse.ArgumentParser(description="处理数据集路径配置")
    
    # 定义输入目录参数
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default='',
        help='输入数据集目录路径'
    )
    
    # 定义输出目录参数
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='',
        help='输出数据集目录路径'
    )
    
    return parser.parse_args()

# ================= 主程序 =================
def main():
    # 1. 获取文件列表
    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    print(f"Found {len(json_files)} JSON files.")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. 初始化 Manager 和 Queue
    manager = Manager()
    log_queue = manager.Queue()

    # 3. 启动日志监听线程 (在主进程中运行)
    listener_thread = threading.Thread(target=log_listener, args=(log_queue,))
    listener_thread.start()

    # 4. 启动多进程池
    print(f"Starting pool with {MAX_WORKERS} workers...")
    
    results_summary = {"total_scanned": 0, "total_kept": 0}

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交任务，注意把 log_queue 传进去
        future_to_file = {
            executor.submit(process_single_json_file, f_path, log_queue): f_path 
            for f_path in json_files
        }

        # 进度条
        progress_bar = tqdm(as_completed(future_to_file), total=len(json_files), unit="file")
        
        for future in progress_bar:
            try:
                file_name, total, kept, status = future.result()
                results_summary["total_scanned"] += total
                results_summary["total_kept"] += kept
                
                # 更新进度条后缀
                progress_bar.set_postfix(kept=results_summary["total_kept"])
                
            except Exception as exc:
                tqdm.write(f"Critical Exception: {exc}")

    # 5. 清理工作
    log_queue.put(None) # 发送结束信号给监听线程
    listener_thread.join() # 等待打印完成

    print("\n" + "="*30)
    print("Processing Complete")
    print(f"Total Scanned: {results_summary['total_scanned']}")
    print(f"Total Kept:    {results_summary['total_kept']}")
    print("="*30)

if __name__ == "__main__":
    args = parse_args()
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir

    main()