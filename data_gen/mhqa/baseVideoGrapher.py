import json
import re
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm


from utils import call_llm_api, video_data


class baseVideoGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph() # 使用 MultiDiGraph，因为两个实体间可能多次发生不同关系
        
        # 1. 核心存储：事件日志
        # 这是一个列表，按时间顺序记录所有发生过的 Triple 事件
        # 结构: [ { "head": A, "rel": R, "tail": B, "desc": "...", "start": 0, "end": 10 }, ... ]
        self.event_log = []
        
        # 2. 已知实体集合 (仅用于去重实体名)
        self.known_entities = set()
        
        # 3. 全局上下文
        self.global_context = ""

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

    def generate_global_summary(self, video_data):
        """Phase 1: 生成全局背景"""
        all_text = " ".join([seg['text'] for seg in video_data['captions']])
        prompt = f"Summarize this video content in 100 words for context:\n{all_text}"
        # print("Generating Global Summary...")
        # self.global_context = call_llm_api(prompt, system_prompt="Summarizer")
        self.global_context = "A tech review video about iPhone 15." # Mock


    def generate_incremental_prompt(self, text, start, end):
        """
        Phase 2: 构建 Prompt
        重点：Context 现在包含最近发生的 Triples 及其描述
        """
        # 获取最近发生的 5 个事件作为短期记忆 (Short-term Memory)
        recent_events = self.event_log[-5:] if self.event_log else []
        
        recent_context_str = ""
        if recent_events:
            recent_context_str = "--- RECENT EVENTS (Context) ---\n"
            for evt in recent_events:
                recent_context_str += f"- [{evt['head']}] {evt['rel']} [{evt['tail']}]: {evt['desc']}\n"
        else:
            recent_context_str = "--- RECENT EVENTS ---\n(None, start of video)"

        # 已知实体列表 (用于名称对齐)
        known_entities_str = ", ".join(list(self.known_entities))


        system_prompt = """
        You are a Precision Relation Extraction Engine for Video Knowledge Graphs.

        **OBJECTIVE**:
        Convert the unstructured video caption into a structured Knowledge Graph format.
        Your goal is **Factual Accuracy** and **Entity Resolution**. Do not editorialize or summarize; extract exactly what is described.

        **TASK**:
        Identify "Interactions" and "States" in the current content.
        For every distinct interaction, output a structured event object.

        **TECHNICAL RULES (Strict Adherence)**:

        1.  **Meta-Data Exclusion (Structural Noise Filter)**:
            *   The graph represents the *content* of the video, not the *medium*.
            *   **DO NOT** create nodes for meaningless entities: "Video", "Clip", "Scene", "Camera", "Footage", "Screen", "Audience", "Narrator".
            *   *Reason*: These are the container, not the data.

        2.  **Entity Canonicalization (Coreference Resolution)**:
            *   You **MUST** resolve pronouns ("he", "she", "it", "they", "this") to their specific referent based on the provided Context.
            *   **Consistency**: Map entities to the names listed in "KNOWN ENTITIES" if they refer to the same object. Do not create duplicate nodes (e.g., do not use "Stark" if "Tony Stark" is already known).

        3.  **Relation Specificity**:
            *   Avoid empty or non-sense relations.

        4.  **Factual Description**:
            *   The "description" field must be a **neutral, standalone statement** describing the triplet's context. Just state the facts.
            
        5. **Entity Diversity**: Avoid focusing solely on the main protagonist. Capture interactions between background entities or objects if relevant.

        **INPUTS**:
        1.  **Global Context**: General topic of the video.
        2.  **Recent Context**: Immediate history for resolving pronouns.
        3.  **Known Entities**: Registry of existing nodes.

        **OUTPUT JSON**:
        {
            "events": [
                {
                    "subject": "Resolved Entity (String)",
                    "relation": "Precise Relation (String)",
                    "object": "Resolved Entity (String)",
                    "description": "Factual context of this specific triplet (String)."
                }
            ]
            "caption": "Observe the changing parts of the video, describe the newly added video clips, and output the extracted entities in concise language."
        }
        """

        user_prompt = f"""
        --- GLOBAL CONTEXT ---
        {self.global_context}

        --- KNOWN ENTITIES (Registry) ---
        {known_entities_str}

        --- RECENT CONTEXT (For Reference) ---
        {recent_context_str}

        --- CURRENT SEGMENT ({start}s - {end}s) ---
        "{text}"

        Extract the structured facts now.
        """
        return user_prompt, system_prompt
    
    def _build_edge_lookup(self, ):
        self.edge_lookup = {}
        # 遍历 self.event_log 中的每一条事件记录，构建映射关系
        for event_record in self.event_log:
            # 取出当前事件的 头实体、尾实体
            head_node = event_record["head"]
            tail_node = event_record["tail"]
            
            # ✅ 核心：双向存入edge_lookup，实现 (u,v) 和 (v,u) 都能查到同一条记录
            self.edge_lookup[(head_node, tail_node)] = event_record
            self.edge_lookup[(tail_node, head_node)] = event_record


    def _update_graph_state(self, data, start, end, **kwargs):
        """
        核心更新逻辑：
        将 Triple + Description + Time 作为一个整体存入 Event Log 和 Graph
        """
        events = data.get("events", [])
        video_clip_caption = data.get("video_clip_caption", [])
        
        for evt in events:
            subj = evt.get("subject")
            rel = evt.get("relation")
            obj = evt.get("object")
            desc = evt.get("description")
            
            if subj and obj and rel:
                # 1. 更新已知实体列表
                self.known_entities.add(subj)
                self.known_entities.add(obj)
                
                # 2. 构建完整的事件记录对象
                event_record = {
                    "head": subj,
                    "rel": rel,
                    "tail": obj,
                    "desc": desc,       # 你的需求：针对这个 triple 的描述
                    "start_time": start,# 你的需求：增加起止时间
                    "end_time": end,
                    "video_clip_caption": video_clip_caption
                }
                event_record.update(kwargs)
                
                # 3. 存入线性日志 (用于 Context 和 导出)
                self.event_log.append(event_record)
                
                # 4. 存入图谱 (MultiDiGraph 支持多条边)
                # 我们把时间、描述都作为边的属性存进去
                self.graph.add_edge(
                    subj, obj, 
                    key=f"{start}_{rel}", # 唯一键，防止覆盖
                    relation=rel,
                    description=desc,
                    start_time=start,
                    end_time=end
                )

    def export_events(self, filename="video_events.json"):
        """导出包含时间、描述的 Triple 列表"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.event_log, f, indent=4, ensure_ascii=False)
        # print(f"Events saved to {filename}")

    def visualize(self, save_path="event_graph.png"):
        """可视化：边上显示关系，鼠标悬停(如果交互式)或打印日志看描述"""
        if self.graph.number_of_nodes() == 0: return
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph, k=2)
        
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', node_size=1500)
        nx.draw_networkx_labels(self.graph, pos, font_weight='bold')
        
        # 绘制边
        # 由于是 MultiGraph，两个节点间可能有多条线，这里简化处理，只画一条
        nx.draw_networkx_edges(self.graph, pos, alpha=0.5, arrowsize=20)
        
        # 边的标签只显示 Relation 动词，不显示长描述，否则图看不清
        edge_labels = {}
        for u, v, k, d in self.graph.edges(keys=True, data=True):
            edge_labels[(u, v)] = d['relation']
            
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color='red')
        
        plt.title("Event-Based Knowledge Graph")
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()
        # plt.show()

    def _mock_response(self, text):
        """仅用于测试的模拟返回"""
        if "iPhone" in text:
            return json.dumps({
                "events": [{
                    "subject": "Host", "relation": "reviews", "object": "iPhone 15",
                    "description": "The host is holding the iPhone 15 and showing its titanium frame to the camera."
                }]
            })
        return '{"events": []}'


# --- 运行 ---
if __name__ == "__main__":
    
    kg = ContextAwareEventGraph()
    kg.process_video_data(video_data)
    kg.export_events() # 检查生成的 JSON，里面会有你要求的所有字段
    kg.visualize()


'''
{
                    "head": subj,
                    "rel": rel,
                    "tail": obj,
                    "desc": desc,       # 你的需求：针对这个 triple 的描述
                    "start_time": start,# 你的需求：增加起止时间
                    "end_time": end,
                    "step": step,
                    "clip_filename": clip_filename,
                    usage_info = {
                        "input_tokens": in_tok, "output_tokens": out_tok, 
                        "cost_cny": round(step_cost, 6), "window_size": len(chat_window)
                    }
                }
'''