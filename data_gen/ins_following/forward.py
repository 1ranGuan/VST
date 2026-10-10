import requests
import json
import re
import time
import os
import base64
import random 
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Any

# ================= 配置区域 =================

API_URL = os.environ.get("VST_LLM_API_URL", "")
API_KEY = os.environ.get("VST_LLM_API_KEY", "")
MODEL_NAME = "gemini-3-flash"
# MODEL_NAME = "gemini-2.5-pro"
# MODEL_NAME = "gemini-3-pro"
MAX_WORKERS = 64  # 根据机器负载调整
MAX_RETRY_TIMES = 8
MAX_HISTORY_TURNS = 4
random.seed(333)

class VideoStreamReasoningGenerator:

    def __init__(self, video_name, video_root, kg_root, output_dir):
        """
        :param video_name: 视频名称 (文件夹名)
        :param video_root: 原始视频存放根目录 (p1_dir)
        :param kg_root: KG数据存放根目录 (p2_dir)
        :param output_dir: 输出目录
        """
        self.video_name = video_name
        self.video_root = video_root
        self.kg_root = kg_root
        self.output_dir = output_dir
        
        self.api_url = API_URL
        self.api_key = API_KEY
        self.model = MODEL_NAME
        
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key
        }

    def load_graph_entities(self):
        """
        加载并格式化实体数据
        """
        # 假设路径结构: kg_root/video_name/kg_gemini3f_{video_name}.json
        graph_json_path = os.path.join(self.kg_root, self.video_name, f"kg_gemini3f_{self.video_name}.json")
        
        if not os.path.exists(graph_json_path):
            print(f"[Warn] KG file not found: {graph_json_path}")
            return []

        try:
            with open(graph_json_path, 'r', encoding='utf-8') as f:
                graph = json.load(f)
        except Exception as e:
            print(f"[Error] Failed to load KG json: {e}")
            return []

        entities_map = {}
        for item in graph:
            # 兼容 step 可能是字符串或数字的情况
            try:
                step = int(item.get('step', -1))
            except:
                continue
                
            if step == -1: continue

            head = item.get('head', 'unknown')
            rel = item.get('rel', 'related to')
            tail = item.get('tail', 'unknown')
            desc = item.get('video_clip_caption', '') # 注意：源数据key可能不同，请根据实际json调整，这里假设是 caption 或者 description

            # entity_desc = f"Head: {head}, Relation: {rel}, Tail: {tail}"
            entity_desc = f"Head: {head}, Relation: {rel}, Tail: {tail}, Discription: {desc}"
            
            if step in entities_map:
                idx = entities_map[step]["entity_num"]
                entities_map[step]["entity_text"] += f"\n- Entity {idx}: {entity_desc}"
                entities_map[step]["entity_num"] += 1
            else:
                entities_map[step] = {
                    "step": step,
                    "start_time": float(item.get("start_time", 0.0)),
                    "end_time": float(item.get("end_time", 0.0)),
                    "entity_num": 1,
                    "entity_text": f"- Entity 0: {entity_desc}"
                }
        
        # 转为列表并排序
        sorted_entities = [
            entities_map[key]
            for key in sorted(entities_map.keys())
        ]
        return sorted_entities

    def load_clip(self, index):
        """读取视频片段并转为Base64"""
        filename = f"part_{index}.mp4"
        # 假设视频结构: video_root/video_name/part_x.mp4
        path = os.path.join(self.video_root, self.video_name, filename)
        
        if not os.path.exists(path):
            # 尝试不带 video_name 子目录的情况，视数据集结构而定
            path = os.path.join(self.video_root, filename)
            if not os.path.exists(path):
                return None
        
        try:
            with open(path, "rb") as video_file:
                return base64.b64encode(video_file.read()).decode("utf-8")
        except Exception as e:
            print(f"Error reading video {path}: {e}")
            return None

    def _call_api(self, prompt, video_b64=None):
        """调用 API"""
        contents = []
        
        
        # 2. 添加视频 (如果有)
        if video_b64:
            contents.append({
                "type": "video",
                "base64": video_b64
            })
        
        # 3. 添加当前 Prompt
        contents.append(prompt)

        payload = {
            "model": self.model,
            "key_id": 1,
            "contents": contents,
            "extra": {
                "temperature": 0.7
            }
        }

        retry_times = 0
        while retry_times <= MAX_RETRY_TIMES:
            try:
                response = requests.post(self.api_url, headers=self.headers, data=json.dumps(payload), timeout=180)
                if response.status_code == 200:
                    resp_json = response.json()
                    # 根据实际 API 返回结构调整
                    if resp_json['code'] == 200:
                        if "choices" in resp_json and len(resp_json["choices"]) > 0:
                            return resp_json["choices"][0]["message"]["content"]
                        elif "data" in resp_json: # 另一种常见格式
                            return resp_json["data"]
                        elif "results" in resp_json: # 你的代码中提到的格式
                            return resp_json["results"][0]
                        else:
                            # 兜底：直接返回 text
                            return str(resp_json)
                    else:
                        print(f"Request Failed: {str(resp_json)}, retry times: {retry_times}")
            except Exception as e:
                print(f"Request Failed: {e}, retry times: {retry_times}")
            print(f"retry times: {retry_times}")
            time.sleep(30)
            retry_times += 1

        return None
    
    def _parse_response(self, response):
        if response:
            # 清洗一下返回的文本，防止包含 markdown 代码块
            response = re.sub(r'^```\w*\s*', '', response).replace('```', '').strip()
        
        response = json.loads(response)
        pos = response["positive_sample"]
        neg = response["negative_sample"]

        return [pos, neg]

    def _construct_prompt(self, ):
        prompt = """You are an expert data generator for Video-LLM training. 
        Your goal is to generate Yes/No QA pairs that test the model's ability to judge **information sufficiency**.

        **Task**:
        Generate 2 QA pairs in JSON format:

        1. **Positive Sample (Yes)**: 
           - Identify an action that is **imminent** or clearly signaled by the visual cues (e.g., a car signaling left -> will turn left; a glass tipping over -> will fall).
           - Question: Ask if we can predict this outcome.
           - Answer: "Yes".

        2. **Negative Sample (No)**:
           - Identify an event that is plausible but **NOT supported** by current visual evidence (e.g., a man walking -> will he meet a friend? (Unknown); a closed box -> is there an apple inside? (Unknown)).
           - Question: Ask if we can predict this specific outcome.
           - Answer: "No".

        **Output Format (JSON)**:
        {
            "positive_sample": {
                "question": "...",
                "answer": "Yes",
                "reasoning": "Brief explanation why it is predictable"
            },
            "negative_sample": {
                "question": "...",
                "answer": "No",
                "reasoning": "Brief explanation why it is NOT predictable"
            }
        }
        """

        return prompt.strip()

    def generate_stream_caption(self):
        """
        主流程
        """
        # 1. 加载实体
        entities = self.load_graph_entities()
        if not entities:
            return {"status": "skipped", "msg": "No entities found"}
        
        # 随机选一个片段
        entity = random.sample(entities, 1)[0]

        step = entity.get("step")
        start_time = entity.get("start_time")
        end_time = entity.get("end_time")
        entity_text = entity.get("entity_text")

        # 2. 初始化结果容器
        results = {
            "video_name": self.video_name,
            "start_time": start_time,
            "end_time": end_time,
            "qa": [] # list of [start, end, text]
        }
        
        # 加载视频
        video_b64 = self.load_clip(step)
        if not video_b64:
            print(f"Warning: Video clip part_{step} missing for {self.video_name}")
            return

        # 构造 Prompt
        prompt = self._construct_prompt()
        
        # 调用 API
        response = self._call_api(prompt, video_b64)
        
        result = self._parse_response(response)
        
        if result == None:
            return 
        
        for qa in result:
            # 存入结果
            results["qa"].append({
                "start_time": start_time,
                "end_time": end_time,
                "question": qa["question"],
                "answer": qa["answer"],
                "reasoning": qa["reasoning"],
            })

        # 4. 保存结果
        if results["qa"]:
            output_file = os.path.join(self.output_dir, f"{self.video_name}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            return {"status": "success", "video": self.video_name}
        else:
            return {"status": "failed", "msg": "No captions generated", "video": self.video_name}

# ==========================================
# 定义单个任务的处理函数 (Top-level function)
# ==========================================

def process_single_video(video_name, p1_dir, p2_dir, output_dir, is_early_stop=False):
    """
    Worker 函数，被进程池调用
    """
    try:
        # 检查输出是否存在，如果存在且不强制覆盖，则跳过
        output_file = os.path.join(output_dir, f"{video_name}.json")
        if os.path.exists(output_file) and is_early_stop:
             return {"status": "skipped", "msg": "Already exists", "video": video_name}

        generator = VideoStreamReasoningGenerator(
            video_name=video_name,
            video_root=p1_dir,
            kg_root=p2_dir,
            output_dir=output_dir
        )
        
        return generator.generate_stream_caption()
        
    except Exception as e:
        # 捕获所有异常，防止进程崩溃
        return {"status": "error", "msg": str(e), "video": video_name}


# ==========================================
# 主程序入口
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="并行视频Caption生成")
    parser.add_argument('--p1_dir', type=str, required=True, help='视频数据集根目录 (包含各个视频文件夹)')
    parser.add_argument('--p2_dir', type=str, required=True, help='KG数据根目录')
    parser.add_argument('--output_dir', type=str, required=True, help='结果输出目录')
    parser.add_argument('--is_early_stop', action="store_true", help='如果结果文件已存在则跳过')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # 参数映射
    VIDEO_ROOT = args.p1_dir
    KG_ROOT = args.p2_dir
    OUTPUT_DIR = args.output_dir
    IS_EARLY_STOP = args.is_early_stop

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 获取任务列表
    if os.path.exists(KG_ROOT):
        # 也可以根据 KG 目录来获取列表，取决于哪个数据更全
        print('scanning diretory...')
        kg_dirs = os.listdir(KG_ROOT)
        # video_name_list = [d for d in kg_dirs if os.path.isdir(os.path.join(KG_ROOT, d))]
        video_name_list = [d for d in kg_dirs if os.path.join(KG_ROOT, d)]

        video_name_list = random.sample(video_name_list, 2000)
        
        print(f"Found {len(video_name_list)} videos. Starting pool with {MAX_WORKERS} workers...")

        # 2. 并行处理
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_video = {
                executor.submit(
                    process_single_video, 
                    v_name, 
                    VIDEO_ROOT, 
                    KG_ROOT, 
                    OUTPUT_DIR, 
                    IS_EARLY_STOP
                ): v_name for v_name in video_name_list
            }

            # 3. 进度条监控
            pbar = tqdm(as_completed(future_to_video), total=len(video_name_list))
            for future in pbar:
                video_name = future_to_video[future]
                try:
                    res = future.result()
                    if res['status'] == 'error':
                        pbar.write(f"[Error] {video_name}: {res['msg']}")
                    elif res['status'] == 'skipped':
                        pass # pbar.write(f"[Skip] {video_name}")
                except Exception as e:
                    pbar.write(f"[Critical] {video_name} crashed: {e}")

        print("All tasks completed.")
    else:
        print(f"Error: Video directory {VIDEO_ROOT} does not exist.")
