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
from multiprocessing import Manager  # 注意：Manager是函数
from multiprocessing.queues import Queue  # 导入Queue基类用于类型注解
from tqdm import tqdm
from typing import Tuple, Optional, Dict, List, Any  # 新增Any用于队列类型

# ================= 配置区域 =================
API_URL = os.environ.get("VST_LLM_API_URL", "")
API_KEY = os.environ.get("VST_LLM_API_KEY", "")
MODEL_NAME = "gemini-3-flash"
MAX_WORKERS = 15  # 并行进程数，根据CPU核数调整
MAX_RETRIES = 3   # API调用最大重试次数
RETRY_DELAY = 1   # 重试延迟（秒）
random.seed(333)

# ================= 工具函数 =================
def validate_config() -> None:
    """校验核心配置项"""
    if not API_URL or not API_KEY:
        raise ValueError("API_URL and API_KEY must be configured!")
    if MAX_WORKERS < 1:
        raise ValueError("MAX_WORKERS must be >= 1!")

# ================= 过滤核心类 =================
class QAFilter:
    """QA数据过滤器：筛选需要依赖视频内容的问答数据"""
    def __init__(self, api_url: str, api_key: str, model: str = "gpt-4o"):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key
        }

    def _call_api_with_retry(self, prompt: str, system_prompt: str = "You are a helpful AI assistant.") -> Optional[str]:
        """带重试机制的API调用"""
        payload = {
            "model": self.model,
            "key_id": 1,
            "contents": prompt.strip(),  # 去除多余空格/换行
            "system_prompt": system_prompt.strip()
        }

        for retry in range(MAX_RETRIES):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    data=json.dumps(payload, ensure_ascii=False),
                    timeout=60
                )
                if response.status_code == 200:
                    resp_json = response.json()
                    if "results" in resp_json and isinstance(resp_json["results"], list) and len(resp_json["results"]) > 0:
                        content = resp_json["results"][0]
                        return self._clean_json_string(content)
                    else:
                        print(f"Warning: Unexpected API response structure (retry {retry+1}/{MAX_RETRIES}): {list(resp_json.keys())}")
                else:
                    print(f"API Error (retry {retry+1}/{MAX_RETRIES}): {response.status_code} - {response.text}")
            except Exception as e:
                print(f"Request Failed (retry {retry+1}/{MAX_RETRIES}): {e}")
            
            time.sleep(RETRY_DELAY * (retry + 1))  # 指数退避

        return None

    def _clean_json_string(self, text: str) -> Optional[str]:
        """清理模型返回的字符串，提取纯JSON内容"""
        if not text:
            return None
        # 移除markdown代码块、多余空格和换行
        text = re.sub(r'^```(json)?\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
        text = text.strip()
        # 移除非JSON前缀/后缀（如模型的解释文本）
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        if json_start != -1 and json_end != 0:
            text = text[json_start:json_end]
        return text

    def _check_world_knowledge_dependency(self, question: str, cot_text: str, answer: str) -> Tuple[bool, str]:
        """
        核心逻辑：检查问题是否依赖视频内容（而非通用常识）
        返回值：(是否保留, 原因)
        """
        # Step 1: 让模型仅基于COT和问题生成回答（模拟无视频场景）
        gen_prompt = f"""
        Based on the given question and chain-of-thought, give a concise direct answer.
        Question: "{question}"
        Chain-of-Thought: "{cot_text}"
        Answer directly without extra explanation.
        """
        generated_answer = self._call_api_with_retry(
            gen_prompt,
            system_prompt="You are a concise AI assistant that answers questions directly."
        )
        
        if not generated_answer:
            return False, "API Error: Failed to generate answer"

        # Step 2: 评估生成答案与真实答案的相似度（判断是否太简单）
        score_prompt = f"""
        Evaluate the semantic equivalence between the candidate answer and ground truth for the given question.
        Input:
        - Question: {question}
        - Ground Truth Answer: {answer}
        - Candidate Answer: {generated_answer}
        
        Scoring Criteria (1-5):
        1: Completely wrong/irrelevant or "I don't know"
        2: Mostly wrong with little correct info
        3: Partially correct (captures keywords but missing core details)
        4: Mostly correct (minor details missing)
        5: Fully equivalent to ground truth
        
        Output ONLY JSON with no extra text:
        {{
            "reasoning": "Brief objective comparison",
            "score": int
        }}
        """
        score_response = self._call_api_with_retry(
            score_prompt,
            system_prompt="You are a strict impartial evaluator. Output ONLY valid JSON."
        )

        if not score_response:
            return False, "API Error: Failed to get score"

        try:
            score_result = json.loads(score_response)
            score = score_result.get('score', 0)
            reasoning = score_result.get('reasoning', 'No reasoning')
            
            # 核心逻辑：score < 4 太难
            if score <= 4 :
                short_ans = generated_answer[:50].replace('\n', ' ') + "..."
                return False, f"Too hard/ cot invalid (score {score}/5). Generated answer: {short_ans}"
            else:
                return True, f"Valid (score {score}/5): {reasoning[:100]}..."
                
        except json.JSONDecodeError:
            return False, f"JSON Parse Error: Invalid score response - {score_response[:100]}..."

    def filter_single_item(self, item: Dict) -> Tuple[bool, str, Optional[Dict]]:
        """处理单个数据项"""
        try:
            # 正确提取text_stream中的文本内容（修复核心错误）
            text_stream = item.get('text_stream', [])
            cot_text = '\n'.join([seg[2] for seg in text_stream if len(seg) >= 3])
            
            qa = item.get('qa', {})
            question = qa.get('question', '')
            answer = qa.get('answer', '')
            cot = qa.get('cot', '')

            # 基础校验
            if not all([question, answer, cot_text]):
                return False, "Missing required fields (question/answer/cot)", None

            # 核心检查：世界知识依赖
            is_valid, reason = self._check_world_knowledge_dependency(question, cot_text, answer)
            return is_valid, reason, item if is_valid else None

        except Exception as e:
            return False, f"Item Processing Error: {str(e)}", None

# ================= 多进程处理函数 =================
# 关键修改：把 Manager.Queue 改为 Any（因为动态创建的队列无法直接注解）
def process_single_json_file(file_path: str, log_queue: Any) -> Tuple[str, int, int, str]:
    """子进程执行：处理单个JSON文件"""
    file_name = os.path.basename(file_path)
    
    def log(msg: str) -> None:
        """封装日志打印"""
        log_queue.put(f"[{file_name}] [{os.getpid()}] {msg}")

    try:
        # 初始化过滤器
        filter_instance = QAFilter(API_URL, API_KEY, model=MODEL_NAME)
        
        # 读取数据
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            log(f"Skipped: Not a list (type: {type(data)})")
            return file_name, 0, 0, "Skipped (non-list)"

        valid_items = []
        total_items = len(data)
        
        # 遍历处理每个item
        for idx, item in enumerate(data):
            is_valid, reason, processed_item = filter_instance.filter_single_item(item)
            if is_valid:
                log(f"Item {idx} ✅ KEEP | {reason}")
                valid_items.append(processed_item)
            else:
                log(f"Item {idx} ❌ DROP | {reason}")

        # 保存有效数据
        if valid_items:
            output_dir = os.environ.get('OUTPUT_DIR')  # 从环境变量获取输出目录
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, file_name)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(valid_items, f, indent=2, ensure_ascii=False)
            log(f"Saved {len(valid_items)}/{total_items} items to {out_path}")
        else:
            log(f"No valid items (0/{total_items})")

        return file_name, total_items, len(valid_items), "Success"

    except Exception as e:
        err_msg = f"Error: {str(e)}"
        log(err_msg)
        return file_name, 0, 0, err_msg

# 关键修改：队列参数改为 Any
def log_listener(queue: Any) -> None:
    """日志监听线程：安全打印日志（不破坏进度条）"""
    while True:
        record = queue.get()
        if record is None:  # 结束信号
            break
        tqdm.write(record)

# ================= 参数解析 =================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QA Dataset Filter (World Knowledge Dependency Check)")
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory with JSON files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for filtered data')
    return parser.parse_args()

# ================= 主程序 =================
def main() -> None:
    # 1. 解析参数并校验
    args = parse_args()
    os.environ['OUTPUT_DIR'] = args.output_dir  # 传递给子进程
    validate_config()

    # 2. 获取文件列表
    json_files = glob.glob(os.path.join(args.input_dir, "*.json"))
    if not json_files:
        print(f"Warning: No JSON files found in {args.input_dir}")
        return
    print(f"Found {len(json_files)} JSON files to process")

    # 3. 初始化日志队列和监听线程
    # 正确用法：先实例化Manager，再创建Queue
    manager = Manager()
    log_queue = manager.Queue()  # 这里才是正确的队列创建方式
    listener_thread = threading.Thread(target=log_listener, args=(log_queue,))
    listener_thread.start()

    # 4. 多进程处理
    results_summary = {"total_scanned": 0, "total_kept": 0}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交任务
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
                progress_bar.set_postfix(
                    total_scanned=results_summary["total_scanned"],
                    total_kept=results_summary["total_kept"]
                )
            except Exception as e:
                tqdm.write(f"Critical error processing file: {e}")

    # 5. 清理资源
    log_queue.put(None)  # 停止日志线程
    listener_thread.join()

    # 6. 打印汇总信息
    print("\n" + "="*50)
    print("Processing Summary")
    print(f"Total items scanned: {results_summary['total_scanned']}")
    print(f"Total items kept:    {results_summary['total_kept']}")
    print(f"Retention rate:      {results_summary['total_kept']/max(results_summary['total_scanned'], 1):.2%}")
    print("="*50)

if __name__ == "__main__":
    main()