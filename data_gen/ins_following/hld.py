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
MAX_WORKERS = 64
MAX_RETRY_TIMES = 8
random.seed(894)

class VideoStreamReasoningGenerator:

    def __init__(self, video_name, video_root, kg_root, output_dir):
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
        graph_json_path = os.path.join(self.kg_root, self.video_name, f"kg_gemini3f_{self.video_name}.json")
        
        if not os.path.exists(graph_json_path):
            graph_json_path = os.path.join(self.kg_root, self.video_name, f"{self.video_name}.json")
            if not os.path.exists(graph_json_path):
                return []

        try:
            with open(graph_json_path, 'r', encoding='utf-8') as f:
                graph = json.load(f)
        except Exception:
            return []

        entities_map = {}
        for item in graph:
            try:
                step = int(item.get('step', -1))
            except:
                continue
            
            if step == -1: continue

            if step not in entities_map:
                entities_map[step] = {
                    "step": step,
                    "start_time": float(item.get("start_time", 0.0)),
                    "end_time": float(item.get("end_time", 0.0))
                }
        
        return list(entities_map.values())

    def load_clip(self, index):
        filename = f"part_{index}.mp4"
        path = os.path.join(self.video_root, self.video_name, filename)
        
        if not os.path.exists(path):
            path = os.path.join(self.video_root, filename)
            if not os.path.exists(path):
                return None
        try:
            with open(path, "rb") as video_file:
                return base64.b64encode(video_file.read()).decode("utf-8")
        except Exception:
            return None

    def _call_api(self, prompt, video_b64=None):
        contents = []
        
        # 1. 添加视频
        if video_b64:
            contents.append({"type": "video", "base64": video_b64})
        
        # 2. 【修正点】直接添加 Prompt 字符串，不要包在 dict 里
        contents.append(prompt)

        payload = {
            "model": self.model,
            "key_id": 1,
            "contents": contents,
            "extra": {"temperature": 0.8}
        }

        retry_times = 0
        while retry_times <= MAX_RETRY_TIMES:
            try:
                response = requests.post(self.api_url, headers=self.headers, data=json.dumps(payload), timeout=180)
                
                if response.status_code == 200:
                    resp_json = response.json()
                    if resp_json.get('code') == 200:
                        if "choices" in resp_json and len(resp_json["choices"]) > 0:
                            return resp_json["choices"][0]["message"]["content"]
                        elif "data" in resp_json:
                            return resp_json["data"]
                        elif "results" in resp_json:
                            return resp_json["results"][0]
                    else:
                        print(f"[API Error] {self.video_name}: {resp_json}")
                else:
                    # 打印 HTTP 错误详情
                    print(f"[HTTP Error] {self.video_name}: {response.status_code} - {response.text[:200]}")

            except Exception as e:
                print(f"[Exception] {self.video_name}: {e}")
            
            time.sleep(2 + retry_times * 2)
            retry_times += 1
        return None
    
    def _parse_response(self, response_text):
        if not response_text: 
            return None
        try:
            cleaned_text = re.sub(r'^```json\s*', '', response_text, flags=re.MULTILINE)
            cleaned_text = re.sub(r'^```\s*', '', cleaned_text, flags=re.MULTILINE)
            cleaned_text = cleaned_text.replace('```', '').strip()
            return json.loads(cleaned_text)
        except Exception:
            return None

    def _construct_prompt(self):
        prompt = """
You are an **Adversarial Question Generator** for Video-LLMs.
Your task is to generate a **Multiple Choice Question (MCQ)** where the correct answer is **"Unable to answer"**.

**Goal**: 
Test if the model hallucinates details that are NOT visible in the video.

**Steps**:
1.  **Identify a Missing Detail**: Find something that is naturally associated with the scene but is **NOT visible, obscured, or out of frame** (e.g., "What brand is the car?" when the logo is blurry; "What is on the table?" when the table is empty or hidden).
2.  **Formulate the Question**: Ask specifically about this missing detail.
3.  **Generate 3 Distractors (Hallucinations)**: Create 3 plausible but **incorrect** answers based on common sense or context.
    *   *Distractor 1 (Co-occurrence)*: An object/detail that usually appears in this scene.
    *   *Distractor 2 (Attribute)*: A specific color, number, or material that fits the context but isn't there.
    *   *Distractor 3 (Action/State)*: An action or state that is likely but not happening.
4.  **Set the Correct Answer**: The correct answer must explicitly state that the information is unavailable.

**Output Format (JSON)**:
{
    "question": "What is the [Missing Detail]?",
    "Distractors": [
        "[Distractor 1 - Co-occurrence Hallucination]",
        "[Distractor 2 - Attribute Hallucination]",
        "[Distractor 3 - Action/State Hallucination]",
    ],
    "answer": "Unable to answer",
    "reasoning": "The video shows [Scene], but the [Detail] is not visible/obscured, so we cannot determine it."
}
"""
        return prompt.strip()

    def generate_stream_caption(self):
        clips = self.load_graph_entities()
        if not clips: return {"status": "skipped", "msg": "No clips found in KG"}
        
        clip_info = random.choice(clips)
        step = clip_info.get("step")
        
        video_b64 = self.load_clip(step)
        if not video_b64: return {"status": "failed", "msg": f"Video part_{step}.mp4 not found"}

        prompt = self._construct_prompt()
        response_text = self._call_api(prompt, video_b64)
        
        if not response_text:
             return {"status": "failed", "msg": "API returned None (Timeout or Error)"}

        result = self._parse_response(response_text)
        
        if not result or "Distractors" not in result:
            return {"status": "failed", "msg": f"Invalid JSON format. Got: {str(result)[:50]}..."}
        
        final_data = {
            "video_name": self.video_name,
            "clip_step": step,
            "start_time": clip_info.get("start_time"),
            "end_time": clip_info.get("end_time"),
            "qa": [result] 
        }

        output_file = os.path.join(self.output_dir, f"{self.video_name}.json")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, indent=4, ensure_ascii=False)
            return {"status": "success", "video": self.video_name}
        except Exception as e:
            return {"status": "error", "msg": f"Write file error: {e}"}

# ==========================================
# Worker & Main
# ==========================================

def process_single_video(video_name, p1_dir, p2_dir, output_dir, is_early_stop=False):
    try:
        output_file = os.path.join(output_dir, f"{video_name}.json")
        if is_early_stop and os.path.exists(output_file):
             return {"status": "skipped", "msg": "Exists", "video": video_name}

        generator = VideoStreamReasoningGenerator(video_name, p1_dir, p2_dir, output_dir)
        return generator.generate_stream_caption()
    except Exception as e:
        return {"status": "error", "msg": f"Worker crash: {e}", "video": video_name}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--p1_dir', type=str, required=True)
    parser.add_argument('--p2_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--is_early_stop', action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.p2_dir):
        print(f"Error: KG Directory {args.p2_dir} does not exist!")
        exit(1)

    kg_dirs = os.listdir(args.p2_dir)
    video_name_list = [d for d in kg_dirs if os.path.isdir(os.path.join(args.p2_dir, d))]
    
    if not video_name_list:
        print("No video folders found in p2_dir.")
        exit(1)

    if len(video_name_list) > 1000:
        video_name_list = random.sample(video_name_list, 1000)
    
    print(f"Starting {len(video_name_list)} tasks with {MAX_WORKERS} workers...")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_video = {
            executor.submit(process_single_video, v, args.p1_dir, args.p2_dir, args.output_dir, args.is_early_stop): v 
            for v in video_name_list
        }

        pbar = tqdm(as_completed(future_to_video), total=len(video_name_list))
        
        success_count = 0
        fail_count = 0

        for future in pbar:
            try:
                res = future.result()
                
                if res['status'] == 'success':
                    success_count += 1
                elif res['status'] == 'skipped':
                    pass
                else:
                    fail_count += 1
                    pbar.write(f"[{res['status'].upper()}] {res.get('video')}: {res.get('msg')}")
                
                pbar.set_description(f"Succ: {success_count} | Fail: {fail_count}")

            except Exception as e:
                pbar.write(f"Critical Main Loop Error: {e}")

    print(f"\nDone. Success: {success_count}, Failed: {fail_count}")
