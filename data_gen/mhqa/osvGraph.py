import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import vertexai
import json
import re
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

from moviepy.editor import VideoFileClip
from vertexai.generative_models import GenerativeModel, Part, Content
import time
import json
import re
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime

import json
import re
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

from baseVideoGrapher import baseVideoGraph

# ================= 配置区域 (保持不变) =================
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
MODEL_NAME = "gemini-3-flash-preview"


# [配置] 价格配置表 (单位: ¥ CNY / 1M Tokens)
PRICING_MAP = {
    "gemini-2.0-flash": {"input": 0.72, "output": 2.88},
    "gemini-2.5-flash": {"input": 2.16, "output": 17.96},
    "gemini-2.5-pro":   {"input": 9.01, "output": 72.09},
    "gemini-3-pro":     {"input": 14.21, "output": 85.28},
    "gemini-3-flash-preview":   {"input": 3.52, "output": 21.12},
    "nano-banana":      {"input": 2.13, "output": 213.68},
    "nano-banana2":     {"input": 14.21, "output": 853.86},
}

# 3. 批处理配置
MAX_HISTORY_TURNS = 1   # 达到4轮强制重置
DEBUG_LIMIT = None       # [Debug] 限制处理的视频数量
MAX_WORKERS = 32         # [多进程] 并行进程数

# 4. 重试与日志配置
MAX_RETRIES = 6        # 最大重试次数
BASE_WAIT_TIME = 10      # 初始等待时间(秒)
LOG_INTERVAL = 4        # 每处理多少个clip记录一次日志 

vertexai.init(project=PROJECT_ID, location=LOCATION)
# =====================================================

class StreamGraphThinker(baseVideoGraph):
    def __init__(self, video_folder, log_file_path, is_visualize=False, is_load_from_log=True, is_load_from_scratch=False):
        self.video_folder = video_folder
        self.log_file_path = log_file_path
        
        # 路径定义
        os.makedirs(os.path.join(os.environ["SAVE_PATH"], self.video_folder), exist_ok=True)
        self.output_json_path = os.path.join(os.environ["SAVE_PATH"], self.video_folder, f"kg_gemini3f_{self.video_folder}.json")
        self.output_graph_path = os.path.join(os.environ["SAVE_PATH"], self.video_folder, f"graph_gemini3f_{self.video_folder}.png") # [新增] 图文件保存路径
        self.is_visualize = is_visualize
        
        # 状态标记
        self.total_step = 0
        self.skip_processing = False
        self.error_video = False
        
        # 费用累计
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        self.is_load_from_log = is_load_from_log
        self.is_load_from_scratch = is_load_from_scratch

        self.graph = nx.MultiDiGraph() # 初始化图
        self.current_timestamp = 0.0   # 全局时间游标 (秒)
        self.known_entities = set() 
        self.event_log = []         
        
        self.total_clips_count = self._count_clips()
        if self.total_clips_count == 0:
            self.error_video = True
            return

        if self.is_load_from_log:
            is_success = self.load_from_log()
            if is_success:
                print('successfully loaded from log.')
                return
            else:
                self.is_load_from_log = False
                self.is_load_from_scratch = True

        if self.is_load_from_scratch:
            self.model = GenerativeModel(MODEL_NAME) 
            with open(self.output_json_path, 'w', encoding='utf-8') as f:
                f.write("[]")
    
    def load_from_log(self):
        """
        核心功能：从self.output_json_path读取保存的event_log，调用父类_update_graph_state重建graph和event_log
        数据无损复用，无需重复调用模型，完美还原所有图谱关系和事件日志
        """
        if not os.path.exists(self.output_json_path):
            self._log(f"日志文件不存在: {self.output_json_path}", level="WARNING")
            return False
        
        # 读取本地保存的日志文件
        with open(self.output_json_path, 'r', encoding='utf-8') as f:
            event_log = json.load(f)
        
        if not event_log or len(event_log) == 0:
            self._log(f"日志文件为空，无法加载", level="WARNING")
            return False
        
        # 清空原有数据，重新初始化
        self.graph = nx.MultiDiGraph()
        self.known_entities = set()
        self.current_timestamp = 0.0
        
        # 遍历读取到的event_log
        for event_record in event_log:
            # 提取_update_graph_state需要的核心参数
            start_time = event_record.get("start_time", 0.0)
            end_time = event_record.get("end_time", 0.0)
            # 组装符合_update_graph_state入参要求的data格式
            data = {
                "events": [{
                    "subject": event_record.get("head"),
                    "relation": event_record.get("rel"),
                    "object": event_record.get("tail"),
                    "description": event_record.get("desc")
                }],
                "video_clip_caption": event_record.get("video_clip_caption")
            }
            # 提取kwargs参数 (step/clip_filename/usage_info等)
            kwargs = {k:v for k,v in event_record.items() if k not in ["head","rel","tail","desc","start_time","end_time"]}
            # 调用父类方法，完美重建图谱和实体集
            self._update_graph_state(data, start_time, end_time, **kwargs)
        
        # 更新全局时间戳为最后一条日志的结束时间
        self.current_timestamp = max([evt.get("end_time",0.0) for evt in event_log])
        self.total_step = max([evt.get("step",0) for evt in event_log]) + 1
        self.skip_processing = True
        return True
            
    def _count_clips(self):
        try:
            full_path = os.path.join(os.environ["VIDEO_PATH"], self.video_folder)
            if not os.path.exists(full_path): return 0
            files = os.listdir(full_path)
            return len([f for f in files if f.startswith("part_") and f.endswith(".mp4")])
        except: return 0

    def _check_if_completed(self):
        if not os.path.exists(self.output_json_path): return False
        try:
            with open(self.output_json_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) >= self.total_clips_count and self.total_clips_count > 0:
                return True
        except: pass
        return False

    def _log(self, message, level="INFO"):
        time_str = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{time_str}] [{level}] [{self.video_folder}] {message}"
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")
        except: pass

    def _calculate_cost(self, usage_metadata):
        if not usage_metadata: return 0.0, 0, 0
        in_tok = usage_metadata.prompt_token_count
        out_tok = usage_metadata.candidates_token_count + getattr(usage_metadata, 'thoughts_token_count', 0)
        prices = PRICING_MAP.get(MODEL_NAME, {"input": 0, "output": 0})
        cost = (in_tok / 1_000_000) * prices["input"] + (out_tok / 1_000_000) * prices["output"]
        return cost, in_tok, out_tok

    def _get_text_safe(self, response):
        try:
            return response.text.strip()
        except:
            try:
                parts = [p.text for p in response.candidates[0].content.parts if p.text]
                return "\n".join(parts).strip()
            except: return ""

    def _clean_and_parse_json(self, raw_text):
        try:
            return json.loads(raw_text)
        except:
            match = re.search(r'```json\s*(.*?)\s*```', raw_text, re.DOTALL)
            if match:
                try: return json.loads(match.group(1))
                except: pass
            match = re.search(r'(\{.*\})', raw_text, re.DOTALL)
            if match:
                try: return json.loads(match.group(1))
                except: pass
            return {"events": []}

    def _get_clip_duration(self, video_path):
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            clip.close() # 记得关闭释放资源
            return duration
        except Exception as e:
            print(f"Error reading video duration: {e}")
            return 0.0

    def load_clip(self, index):
        filename = f"part_{index}.mp4"
        path = os.path.join(os.environ["VIDEO_PATH"], self.video_folder, filename)
        if not os.path.exists(path): return None, None # 修改返回值，同时返回路径
        with open(path, "rb") as f:
            return Part.from_data(f.read(), mime_type="video/mp4"), path

    def build_kg_prompt(self, step_index, start_time, end_time):
        # [修改] Prompt 中加入具体的时间段信息
        known_entities_list = list(self.known_entities)
        if len(known_entities_list) > 50: 
            known_entities_list = known_entities_list[-50:]
        known_entities_str = ", ".join(known_entities_list)

        prompt = f"""
            [SYSTEM ROLE]
            You are a **Visual Scene Analyst** specializing in dense scene graph generation.
            Your goal is to map **ALL** physical relationships in the video segment, not just human actions.

            [CURRENT TIMELINE]
            Segment #{step_index}: {start_time:.2f}s to {end_time:.2f}s.

            [CONTEXTUAL DATA]
            1. **Entity Registry**: {known_entities_str}

            [CRITICAL VISUAL EXTRACTION RULES]
            1. **🚫 NO "HUB-AND-SPOKE" BIAS (CRITICAL)**:
            *   **Do not** make the human protagonist the subject of every relation.
            *   **FAILURE CASE**: A graph where all arrows point from "Person" to objects is **UNACCEPTABLE**.
            *   **SUCCESS CASE**: You must extract edges where **neither** the subject nor the object is a person.

            2. **Object-to-Object Relationships (MANDATORY)**:
            *   Look for **Spatial Relations**: Is the "Frying Pan" *on* the "Stove"? Is the "Sausage" *inside* the "Pan"?
            *   Look for **Containment**: Does the "Bottle" *contain* "Oil"?
            *   Look for **Passive Interactions**: Is the "Steam" *rising from* the "Pot"?
            *   **Example**: Instead of just (Person, holds, Pan), ALSO extract (Pan, contains, Food) and (Pan, sits on, Stove).

            3. **Visual Entity Identification**:
            *   Identify objects even if the person is not touching them right now.
            *   Use specific visual descriptors (e.g., "Silver Metal Pot", "Red Gas Canister").
            *   **NO PRONOUNS**.

            4. **Action & State Verbs**:
            *   Use active verbs for humans (e.g., "lifts", "pours").
            *   Use spatial/state verbs for objects (e.g., "contains", "rests on", "next to", "inside").

            5. **Description**:
            *   Describe the scene layout and object states, not just the human's movement. (e.g., "The sausages are sizzling inside the hot pan while it sits on the lit stove.")

            [OUTPUT FORMAT]
            Return ONLY a JSON object:
            {{
                "events": [
                    {{
                        "subject": "Entity A (Can be an object)",
                        "relation": "Interaction or Spatial Relation",
                        "object": "Entity B (Can be an object)",
                        "description": "Context of this specific relationship."
                    }}
                ],
            }}
            """



        return prompt

    def save_to_json(self, ):
        # [修改] 保存时加入时间戳
        try:
            if os.path.exists(self.output_json_path):
                with open(self.output_json_path, 'r', encoding='utf-8') as f:
                    current_data = json.load(f)
            else:
                current_data = []
            
            current_data = current_data + self.event_log
            with open(self.output_json_path, "w", encoding="utf-8") as f:
                json.dump(current_data, f, ensure_ascii=False, indent=4)
        except Exception:
            pass

    def run(self):
        if self.skip_processing: return 0.0

        self._log(f"Start KG Extraction. Total clips: {self.total_clips_count}", level="START")
        
        chat_window = deque(maxlen=MAX_HISTORY_TURNS)
        self.current_timestamp = 0.0 # 重置时间

        while True:
            clip_filename = f"part_{self.total_step}.mp4"
            # [修改] load_clip 现在返回 (part, path)
            video_part, video_path = self.load_clip(self.total_step)
            
            if video_part is None:
                self._log(f"Finished. Final Cost: ¥{self.total_cost:.4f}", level="DONE")
                # [可选] 结束时保存 Graph 对象
                # nx.write_gpickle(self.graph, self.output_graph_path) 
                break

            # 计算当前 Clip 的起止时间
            clip_duration = self._get_clip_duration(video_path)
            start_time = self.current_timestamp
            end_time = start_time + clip_duration
            
            # 1. 构建 Prompt (传入时间)
            prompt_text = self.build_kg_prompt(self.total_step, start_time, end_time)
            
            # 2. 构建 API History
            history_for_api = []
            for item in chat_window:
                # user_content = Content(role="user", parts=[item["video"], Part.from_text(item["prompt"])])
                user_content = Content(role="user", parts=[item["video"]])
                history_for_api.append(user_content)
                model_content = Content(role="model", parts=[Part.from_text(item["response_text"])])
                history_for_api.append(model_content)

            # 3. 新建 Chat Session
            current_chat = self.model.start_chat(history=history_for_api)

            retry_count = 0
            success = False
            
            while retry_count < MAX_RETRIES:
                try:
                    response = current_chat.send_message([video_part, prompt_text])
                    response_text = self._get_text_safe(response)
                    
                    step_cost, in_tok, out_tok = self._calculate_cost(response.usage_metadata)
                    self.total_cost += step_cost
                    self.total_input_tokens += in_tok
                    self.total_output_tokens += out_tok
                    
                    # 4. 解析 JSON
                    parsed_json = self._clean_and_parse_json(response_text)
                    
                    # 5. 更新图状态 (传入时间)
                    usage_info = {
                        "input_tokens": in_tok, "output_tokens": out_tok, 
                        "cost_cny": round(step_cost, 6), "window_size": len(chat_window)
                    }
                    self._update_graph_state(parsed_json, start_time, end_time, 
                                            step = self.total_step,
                                            clip_filename = clip_filename,
                                            usage_info = usage_info
                                            )

                    # 6. 保存 JSON 记录
                    self.save_to_json()
                    
                    chat_window.append({
                        "video": video_part,
                        "prompt": prompt_text,
                        "response_text": response_text
                    })

                    success = True
                    if self.total_step > 0 and self.total_step % LOG_INTERVAL == 0:
                        events_count = len(parsed_json.get("events", []))
                        self._log(f"Step {self.total_step} ({start_time:.1f}s-{end_time:.1f}s): Extracted {events_count} events. Cost: ¥{step_cost:.4f}", level="INFO")
                    break 

                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "Resource exhausted" in error_str or "Internal" in error_str:
                        retry_count += 1
                        wait_time = min(BASE_WAIT_TIME * (2 ** (retry_count - 1)), 60)
                        time.sleep(wait_time)
                    else:
                        self._log(f"Fatal Error at step {self.total_step}: {e}", level="ERROR")
                        print(f"Fatal Error at step {self.total_step}: {e}")
                        break
            
            if success:
                self.total_step += 1
                # [核心逻辑] 更新时间游标
                self.current_timestamp = end_time
                time.sleep(0.5)
                # print(f'building graph: the {self.total_step} step')

                # if self.total_step>=3:
                #     break
            else:
                print('failure')
                break
        if self.is_visualize:
            self.visualize(self.output_graph_path)

        return self.total_cost

# ================= 外部调用包装 (保持不变) =================
def process_single_folder(folder_name, log_file_path):
    try:
        full_path = os.path.join(os.environ["VIDEO_PATH"], folder_name)
        if not os.path.exists(full_path): return None, 0.0
        
        # 使用新的 StreamGraphThinker
        thinker = StreamGraphThinker(folder_name, log_file_path)
        if thinker.error_video: return None, 0.0
        
        video_cost = thinker.run()
        return folder_name, video_cost
    except Exception as e:
        return None, 0.0

# ================= 主程序 (保持不变) =================
if __name__ == "__main__":
    # ... (这部分代码与你之前提供的完全一致，只需确保调用的是 process_single_folder) ...
    # 为了完整性，这里简写主程序逻辑，实际使用时请复制你之前的 main block
    
    # 读取过滤列表
    filted_file = os.environ["FILTED_VM_1224"]
    with open(filted_file, 'r', encoding='utf-8') as f:
        all_items = json.load(f)
    
    target_folders = all_items[:DEBUG_LIMIT] if DEBUG_LIMIT else all_items
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("process_vm", exist_ok=True)
    log_filename = f"process_vm/kg_run_monitor_{timestamp}.log"
    
    print(f"🚀 Starting KG Extraction on {len(target_folders)} videos...")
    

    total_run_cost = 0.0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_folder, f, log_filename): f for f in target_folders}
        
        for future in tqdm(as_completed(futures), total=len(target_folders)):
            folder, cost = future.result()
            if folder:
                total_run_cost += cost
                tqdm.write(f"💰 {folder}: ¥{cost:.4f}")
    
    print(f"✅ All Done. Total Cost: ¥{total_run_cost:.4f}")
