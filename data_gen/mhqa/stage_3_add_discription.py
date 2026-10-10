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
MAX_WORKERS = 128  # 根据机器负载调整
MAX_RETRY_TIMES = 8
MAX_HISTORY_TURNS = 4
random.seed(333)

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

                # print("#"*100)
                # print( text)
                print("repeated text: ", text[i : i + window_size])
                return True
        
        seen_hashes.add(current_hash)
        
    return False


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

    def _call_api(self, previous_context, prompt, video_b64=None):
        """调用 API"""
        contents = []
        
        # 1. 添加历史上下文
        if previous_context:
            contents.append(previous_context)
        
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
                        if "results" in resp_json: # 你的代码中提到的格式
                            response = resp_json["results"][0]
                            text = previous_context + response
                            if not check_repetition_rolling_hash(text, 60):
                                return response
                            else:
                                print('repeated response')
                                if retry_times >= MAX_RETRY_TIMES:
                                    return response
                    else:
                        print(f"Request Failed: {str(resp_json)}, retry times: {retry_times}")
            except Exception as e:
                print(f"Request Failed: {e}, retry times: {retry_times}")
            print(f"retry times: {retry_times}")
            time.sleep(30)
            retry_times += 1

        return None

    def _construct_prompt(self, current_time_range, entity_text, is_initial_segment):
        """
        构造 Prompt
        """
        if is_initial_segment:
            # === 策略 A: 初始化 Prompt (建立场景) ===
            prompt = f"""
Current Video Segment: {current_time_range}

**Task**: 
This is the start of a video. Provide a description of the opening scene. Establish the setting, background, visible entities, and their initial appearance/state.

**Context & Entities**:
Detected Entities: {entity_text}
(Describe these entities FULLY to establish who/what is in the scene.)

**Strict Constraints**:
1. **Establish Context**: Clearly describe WHERE the scene takes place and WHO is present.
2. **Detailed Appearance**: Since this is the start, you MAY describe the visual appearance (clothing, colors, lighting) of key entities.
3. **Initial Action**: Describe the very first action occurring in this segment.
4. **Tone**: Objective and descriptive, setting the stage for the rest of the video.

**Output**:
"""
        else:
            prompt = f"""
Current Video Segment: {current_time_range}

**Task**: 
Provide a incremental update of the visual scene/action/details in this segment. 
**Focus on Progress (Delta)** of visual elements.

**Context & Entities**:
Focused Entities: {entity_text}
(Integrate these naturally if they are actively involved in the current action. No full repetition each time; pronouns prohibited.)

**Strict Constraints**:
1. **Focus on Dynamics**: Describe WHAT is happening now with details. If the scene is static, be very concise (do not repeat using "remain..."). If the scene is static, be very concise.
2. **NO restate**: Do NOT restate the action/scene/details from the previous segment's end. Skip the action stated previously or state the changes. NEVER repeat.
3. **Minimalist Reference**: For entities already present/mentioned, use pronouns or minimal descriptors. Do NOT re-describe their appearance unless it changes.
4. **Language Variety**: Avoid repetitive sentence structures (e.g., do not start every sentence with "The man...").
5. **No Redundancy**: Do not repeat the timestamp or information from the previous context. Use pronouns naturally. No re-descriptions.

**Output**:
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

        # 2. 初始化结果容器
        results = {
            "video_name": self.video_name,
            "captions": [] # list of [start, end, text]
        }
        
        previous_history = "Previous Context:\n[Start of Video]\n"
        history_text = []
        
        # 3. 遍历每个片段
        for i, entity in enumerate(entities):
            step = entity.get("step")
            start_time = entity.get("start_time")
            end_time = entity.get("end_time")
            entity_text = entity.get("entity_text")
            
            # 加载视频
            video_b64 = self.load_clip(step)
            if not video_b64:
                print(f"Warning: Video clip part_{step} missing for {self.video_name}")
                continue

            current_time_range = f"{start_time:.1f}s - {end_time:.1f}s"
            
            # 构造 Prompt
            prompt = self._construct_prompt(current_time_range, entity_text, i==0)
            
            # 调用 API
            # 注意：previous_history 会作为 contents 的第一个元素传入
            caption = self._call_api(previous_history+''.join(history_text[:MAX_HISTORY_TURNS]), prompt, video_b64)
            
            if caption:
                # 清洗一下返回的文本，防止包含 markdown 代码块
                caption = re.sub(r'^```\w*\s*', '', caption).replace('```', '').strip()
                
                # 存入结果
                results["captions"].append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "caption": caption
                })
                
                # 更新历史上下文，供下一段使用
                # 限制历史长度，防止 token 溢出（可选）
                history_text.append(f"\n[Time {current_time_range}]: {caption}")
                
                # 简单的日志
                # print(f"[{self.video_name}] Step {step} done.")
            else:
                print(f"[{self.video_name}] Step {step} failed to generate.")

        # 4. 保存结果
        if results["captions"]:
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
        print(f"Found {len(video_name_list)} videos in total.")

        if IS_EARLY_STOP:
            # 检查输出是否存在，如果存在且不强制覆盖，则跳过
            output_files = os.listdir(OUTPUT_DIR)
            # video_name_list = [d for d in kg_dirs if os.path.isdir(os.path.join(KG_ROOT, d))]
            processed_video_names = [d.split(".")[0] for d in output_files]
            video_name_list = list( set(video_name_list) - set(processed_video_names))

        


        print(f"Found unprocessed {len(video_name_list)} videos. Starting pool with {MAX_WORKERS} workers...")

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
