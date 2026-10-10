import networkx as nx
import json
import random
import re
from collections import deque

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
from google.api_core import retry

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


from utils import call_llm_api, video_data
from osvGraph import StreamGraphThinker

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
MAX_HISTORY_TURNS = 4   # 达到4轮强制重置
DEBUG_LIMIT = None       # [Debug] 限制处理的视频数量
MAX_WORKERS = 32         # [多进程] 并行进程数

# 4. 重试与日志配置
MAX_RETRIES = 6        # 最大重试次数
BASE_WAIT_TIME = 10      # 初始等待时间(秒)
LOG_INTERVAL = 4        # 每处理多少个clip记录一次日志 
EXISTING_PROCESSED_FILE = os.environ.get("EXISTING_PROCESSED_FILE", "process_vm/qa_gen_processed_list.json" )


vertexai.init(project=PROJECT_ID, location=LOCATION)

class VideoGraphQAGenerator:
    def __init__(self, video_folder, log_file_path, qa_num=3, output_folder="qa_dataset", is_visualize=False):
        """
        :param video_folder: 存放原始视频文件的文件夹
        :param log_file_path: 之前 StreamGraphThinker 生成的日志文件 (JSONL 或 JSON)
        :param qa_num: 每个视频生成的 QA 数量
        :param output_folder: 结果保存路径
        """
        self.video_folder = video_folder
        self.log_file_path = log_file_path
        self.qa_num = qa_num
        self.output_folder = output_folder
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 1. 初始化LLM
        system_instruction = self._get_system_prompt()
        # retry_policy = retry.Retry(total=60, initial=2, multiplier=2)
        self.model = GenerativeModel(MODEL_NAME, system_instruction=system_instruction)
        # self.model._generation_config["retry"] = retry_policy
        # self.model._generation_config["timeout"] = 120.0

        # 2. 重建图谱 & 加载日志
        self.thinker = StreamGraphThinker(video_folder, log_file_path, is_visualize=is_visualize)
        video_cost = self.thinker.run()
        self.thinker._build_edge_lookup()
        self.buidl_graph_cost = video_cost
        self.data_entry = self.thinker.event_log
        self.graph = self.thinker.graph

    def _check_graph_diameter(self):
        """计算图的直径 (最长最短路径)，判断是否值得生成 QA"""
        if self.graph.number_of_nodes() < 2:
            return 0
            
        # NetworkX 的 diameter 算法比较慢，且要求图是连通的。
        # 对于非连通图，我们计算最大连通分量的直径。
        undirected_g = self.graph.to_undirected()
        if nx.is_connected(undirected_g):
            diameter = nx.diameter(undirected_g)
        else:
            # 获取所有连通分量
            components = nx.connected_components(undirected_g)
            max_dia = 0
            for c in components:
                subg = undirected_g.subgraph(c)
                if subg.number_of_nodes() > 1:
                    d = nx.diameter(subg)
                    if d > max_dia:
                        max_dia = d
            diameter = max_dia
            
        return diameter

    def extract_reasoning_paths_dfs(self, max_depth=8, max_paths=5, similarity_threshold=0.2):
        """基于DFS挖掘长路径 - 修复高比例重复问题，保证路径多样性"""
        search_graph = self.graph.to_undirected()
        paths = []
        nodes = list(search_graph.nodes())
        random.shuffle(nodes)

        def calculate_path_similarity(path1, path2):
            set1 = set(path1)
            set2 = set(path2)
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0.0

        def is_path_unique(new_path):
            for exist_path in paths:
                sim = calculate_path_similarity(new_path, exist_path)
                if sim >= similarity_threshold:
                    return False
            return True

        for source in nodes:
            if len(paths) >= max_paths:
                break
            current_max_depth = random.randint(2, max_depth)
            stack = [(source, [source])]

            while stack and len(paths) < max_paths:
                curr_node, current_path = stack.pop(0)  # pop(0)优先探索分支，减少重复
                if len(current_path) > current_max_depth + 1:
                    continue

                if len(current_path) >= current_max_depth and len(current_path) >= 4:
                    if is_path_unique(current_path):
                        paths.append(current_path)
                    continue

                neighbors = list(search_graph.neighbors(curr_node))
                random.shuffle(neighbors)
                for neighbor in neighbors:
                    if neighbor not in current_path:
                        stack.append((neighbor, current_path + [neighbor]))

        return paths[:max_paths]

    def load_clip(self, index):
        filename = f"part_{index}.mp4"
        path = os.path.join(os.environ["VIDEO_PATH"], self.video_folder, filename)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return Part.from_data(f.read(), mime_type="video/mp4")

    def _build_step_text_content(self, index, entry, u, v):
        """
        辅助函数：构建单步的文本描述信息
        """
        start = entry.get('start_time', 0.0)
        end = entry.get('end_time', 0.0)
        
        # 提取当前边 (u, v) 相关的具体事件信息
        events = entry.get('extracted_events', [])
        relevant_event = None
        for evt in events:
            # 匹配当前路径上的边 (正向或反向)
            if (evt.get('subject') == u and evt.get('object') == v) or \
               (evt.get('subject') == v and evt.get('object') == u):
                relevant_event = evt
                break
        
        # 如果没找到特定的边，就用 entry 里第一个事件兜底，或者显示 Generic Info
        if relevant_event:
            triplet = f"({relevant_event.get('subject')}, {relevant_event.get('relation')}, {relevant_event.get('object')})"
            desc = relevant_event.get('description', 'No description available.')
        else:
            triplet = f"({u}, related_to, {v})"
            desc = "Interaction between these entities."

        text_content = f"""
            --- Step {index} ---
            **Time Interval**: {start:.2f}s - {end:.2f}s
            **Core Interaction**: {triplet}
            **Visual Description**: {desc}
            **Video Segment**: (See the video clip below)
            """
        return text_content

    def _get_system_prompt(self):
        """
        构建 System Prompt
        """
    def _get_system_prompt(self):
        """
        构建 System Prompt (V2 - High Quality)
        """
        # 1. 动态采样推理维度 (保持多样性)
        all_dimensions = [
            "- **Cross-Modal Analysis**: Contrast visual actions with audio/narrative context. Are they aligned or contradictory?",
            "- **Causal Dependency**: Explain how an early action physically caused a later outcome.",
            "- **Temporal Logic**: Analyze the sequence order, duration, or frequency of actions.",
            "- **State Tracking**: Trace the transformation of an object or emotion throughout the video.",
            "- **Counterfactuals**: Hypothesize a realistic alternative outcome based on the logic shown.",
            "- **Predictive Logic**: Predict the immediate next consequence based on the observed trend."
        ]

        all_dimensions = [
            # === 1. Perception & Detail (对应 OCR, STU, ATR, OJR) ===
            "- **Visual Text Contextualization**: Integrate embedded text (OCR) within the scene (e.g., signs, screens, documents) with visual actions to explain the context.",
            "- **Spatial Dynamics**: Analyze the relative position, depth, or spatial relationship changes between objects across different segments.",
            "- **Attribute Evolution**: Track fine-grained changes in object states (e.g., color, shape, cleanliness, completeness) to infer what happened in between.",
            
            # === 2. Temporal & Sequence (对应 REC, SSR, ASI, ACR) ===
            "- **Frequency & Pattern Recognition**: Count specific repetitive actions or object appearances across non-contiguous segments to determine frequency.",
            "- **Procedural Reconstruction**: Identify the logical order of a complex task (e.g., cooking, assembly) and determine missing or implied steps.",
            "- **Action Duration Analysis**: Compare the relative duration of different actions to judge their significance or difficulty.",

            # === 3. Cognition & Reasoning (对应 EPM, HLD, FPD, CRR) ===
            "- **Episodic Retrieval**: Link a current visual cue back to a specific past detail (e.g., 'Where was this object first seen?') to test memory.",
            "- **Logical Consistency Verification**: Detect visual contradictions or impossible physical transitions between segments (Hallucination Detection).",
            "- **Causal Dependency**: Explain how an early action physically caused a later outcome.",
            "- **Goal Inference**: Deduce the ultimate intent or final goal of the actor based on a sequence of subtle clues (Clues Reveal).",
            "- **Predictive Logic**: Predict the immediate next consequence or the final state based on the observed trend.",
            "- **Counterfactuals**: Hypothesize a realistic alternative outcome if a key action in the sequence had been different."
        ]

        # 随机选 2-3 个维度，避免 Prompt 过长且保持新鲜感
        selected_dims = random.sample(all_dimensions, random.randint(2, 3))
        dimensions_str = '\n'.join(selected_dims)

        prompt = f"""
            You are a **Cognitive Video Intelligence Engine**.
            Your task is to synthesize a **Practical Deep Reasoning Question-Answer Pair** based on the provided visual information from video segments and the **Event Reasoning Path**.

            **=== CORE DEFINITION (MUST FOLLOW FIRST) ===**
            1.  **Event Reasoning Path Nature**: The provided video segments are ordered by **logical reasoning relevance**, NOT by their original chronological time sequence in the video.
            2.  **Time Reference Rule**: All time mentions in your output must be based on the **explicit time interval of each node in the reasoning path** (e.g., "around 10-20s", "at the 45-second mark"), NOT the order of segments in the path.
            3.  **Multi-hop Foundation**: The reasoning must rely on **logical connections between segments** (e.g., causal, temporal, conditional) in the path, even if their chronological order is non-linear.

            **=== MANDATORY INSTRUCTIONS ===**
            1.  **Strict Multi-hop Reasoning**:
                The question MUST require integrating information from **multiple video segments in the reasoning path**. It is impossible to answer correctly by analyzing any single segment alone.

            2.  **Natural Language Constraint (CRITICAL)**:
                - DO NOT use the words "Step", "Clip index", "Path node" in the question, answer, or reasoning chain.
                - Refer to segments using **their time intervals** (e.g., "from 100s to 135s", "around the 45-second mark") or **event descriptions** (e.g., "when the person handled raw sausages", "while the pan was heating").

            3.  **Reasoning Dimensions**:
                Construct the reasoning chain using one or more of the following logic types:
                {dimensions_str}

            4.  **Practicality Requirement**:
                The question must be meaningful for understanding the visual content of the video's **narrative, intent, physical logic or any perception details**.

            **=== OUTPUT FORMAT (JSON ONLY, NO EXTRA TEXT) ===**
            {{
                "question": "A multi-hop question linking multiple events from different time intervals in the reasoning path. (natural language, no 'Step' references, no explicit time intervals).",
                "cot": "Chain-of-Thought: Explicitly cite the time interval and visual evidence of each relevant segment, then explain their logical connection (e.g., causal, temporal, state-tracking) to derive the answer.",
                "answer": "A very short, concise, direct answer based on the reasoning chain, with no references to step/clip numbers.",
                "reasoning_type": "The primary logic dimension used (must match one of the provided types)."
            }}
            """
        return prompt.strip()


    def generate_qa_with_mllm(self, path, path_entries):
        """
        调用 MLLM 生成 QA (图文交错模式)。
        :param path: 节点列表 [A, B, C...]
        :param path_entries: 对应的详细 Log Entry 列表，每个元素是指定格式的字典
        """
        # ========== 1. 初始化 Contents 列表 + 第一部分：拼接【可视化路径字符串】给LLM ==========
        # 核心修改：生成 path[0]->path[1]->path[2] 格式的路径文本，作为给LLM的前置信息
        path_str = " -> ".join(path)
        contents = [
            f"### Event Reasoning Path: {path_str}\n",
            f"Please analyze the following event chain in strict chronological order, combined with video clips:\n\n"
        ]
        
        # ========== 2. 构建图文交错的内容：先文本（三元组核心信息） → 后视频片段 ==========
        # 遍历路径上的每一步 (边) path长度为 N，边数为 N-1
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i+1]
            entry = path_entries[i]
            
            # A. 核心提取：只取字典里指定的7个关键字段，精准给到LLM
            step = entry.get('step', '')
            head = entry.get('head', '')
            rel = entry.get('rel', '')
            tail = entry.get('tail', '')
            desc = entry.get('desc', 'No description')
            start_time = entry.get('start_time', 0.0)
            end_time = entry.get('end_time', 0.0)

            # B. 构建标准化的【三元组核心信息文本】，格式规整，LLM友好
            text_part = f"""
                --- Step {i+1}: {u} -> {v} ---
                ✅ Triple Info: ({head}, {rel}, {tail})
                ⏰ Time Range: {start_time:.2f}s ~ {end_time:.2f}s
                📝 Step Index: {step}
                """
            # C. 加载当前step对应的视频片段
            video_part = self.load_clip(step)
            
            # D. 先追加文本，再追加视频（核心顺序不变，图文交错）
            contents.append(text_part)
            if video_part is not None:
                contents.append(video_part)
            else:
                print(f"Skipping step {i+1} (path: {u}->{v}) due to missing video part.")
                continue
                
        # ========== 3. 添加最后的触发指令，强制指定输出格式 ==========
        contents.append("""
                    Based on the above reasoning path + triple information + video clips, generate 1 high-quality multi-hop reasoning QA pair.
                    Strictly output in valid JSON format only, no extra text, no explanation, no markdown.
                """)

        # ========== 4. 调用模型 + 超时重试 + 异常捕获 ==========
        try:
            response = self.model.generate_content(contents)
            
            print(f"✅ MLLM Response generated for path: {path_str}")
            res = self._clean_json_string(response.text)
            return res
            
        except Exception as e:
            print(f"❌ MLLM Generation Failed for path {path_str}: {str(e)}")
            return None


    def _clean_json_string(self, json_str):
        """清洗 JSON"""
        try:
            return json.loads(json_str)
        except:
            match = re.search(r'```json\s*(.*?)\s*```', json_str, re.DOTALL)
            if match: return json.loads(match.group(1))
            match = re.search(r'(\{.*\})', json_str, re.DOTALL)
            if match: return json.loads(match.group(1))
            return {"events": []}

    def run(self):
        """主执行函数"""
        # 1. 检查图直径
        diameter = self._check_graph_diameter()
        print(f"Graph Diameter: {diameter}")
        
        if diameter < 2:
            print("❌ Graph is too shallow (diameter < 3). Skipping video.")
            return []
            
        print("✅ Graph is deep enough. Proceeding to path extraction.")
        
        # 2. 挖掘路径
        # 注意：这里的路径是节点列表 [A, B, C, D]
        raw_paths = self.extract_reasoning_paths_dfs(max_depth=4, max_paths=self.qa_num)
        print(f"Found {len(raw_paths)} potential reasoning paths.")
        
        final_results = []
        
        for idx, path in enumerate(raw_paths):
            print(f"\n--- Processing Path #{idx+1}: {path} ---")
            
            # 3. 回溯路径对应的 Log Entries (获取时间戳和元数据)
            path_entries = []
            valid_path = True
            
            # 遍历路径上的每一条边 (u, v)
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                
                # 尝试查找对应的 entry
                entry = self.thinker.edge_lookup.get((u, v)) or self.thinker.edge_lookup.get((v, u))
                if not entry:
                    print(f"Warning: Missing log entry for edge {u}-{v}. Skipping path.")
                    valid_path = False
                    break
                
                # 避免重复添加同一个 entry (如果多个边在同一个 clip 里)
                if not path_entries or path_entries[-1] != entry:
                    path_entries.append(entry)
            
            if not valid_path or not path_entries:
                continue
                
            # 5. 生成 QA
            qa_pair = self.generate_qa_with_mllm(path, path_entries)
            
            if qa_pair:
                # 6. 组装最终结果
                result_item = {
                    "video_name": self.video_folder+'.mp4', 
                    "path_nodes": path,
                    "path_entries": path_entries,
                    "qa": qa_pair
                }
                final_results.append(result_item)
        
        # 7. 保存结果
        if final_results:
            save_path = os.path.join(self.output_folder, f"{self.video_folder}.json")
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            print(f"🎉 Saved {len(final_results)} QA pairs to {save_path}")
            
        return final_results

if __name__ == "__main__":
    # 配置
    filted_file = os.environ["FILTED_VM_1224"]
    with open(filted_file, 'r', encoding='utf-8') as f:
        all_items = json.load(f)
    
    target_folders = all_items[:DEBUG_LIMIT] if DEBUG_LIMIT else all_items
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("process_vm", exist_ok=True)
    log_filename = f"process_vm/kg_run_monitor_{timestamp}.log"
    
    print(f"🚀 Starting KG Extraction on {len(target_folders)} videos...")


    
   

    VIDEO_FOLDER = target_folders[0]
    LOG_FILE = log_filename
    OUTPUT_FOLDER = os.environ['OUTPUT_FOLDER']

    generator = VideoGraphQAGenerator(
        video_folder=VIDEO_FOLDER,
        log_file_path=LOG_FILE,
        qa_num=3,
        output_folder=OUTPUT_FOLDER, 
        is_visualize=True
    )
    
    generator.run()
