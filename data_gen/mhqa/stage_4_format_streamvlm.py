import os
import json
import argparse
from typing import List, Dict, Any

# ===================== 核心配置区（可直接修改对应路径） =====================

# 系统固定prompt模板，直接复制使用
SYSTEM_PROMPT_TEMPLATE = """[System]
You are a Streaming Video Analyst.

**Streaming Thinking Rules:**
1. **Update Only**: Read <previous_thought>. Only record **new** clues from the current segment relevant to <problem>. Do not repeat history.
2. **Wait for End**: Do not provide the final answer until the video stream is complete. Currently, just accumulate evidence

<problem>
{prompt}
</problem>

<previous_thought>
{memory}
</previous_thought>"""

TEMPLATE_FINAL_BOXED = """You are presented with a problem and a previous memory derived from watching a video. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

{PromptFinal}

Your answer:
"""

class JsonDataFormatter:
    def __init__(self):
        # 初始化加载【视频名称-路径】映射字典，全局复用
        self.name2path = self._load_name2path_mapping()
        # 初始化历史文本记忆，用于拼接previous字段
        self.history_text = ""

    def _load_name2path_mapping(self) -> Dict[str, str]:
        """加载视频名称到视频路径的映射json文件"""
        try:
            with open(NAME2PATH_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"映射文件不存在: {NAME2PATH_JSON}")
        except json.JSONDecodeError:
            raise ValueError(f"映射文件格式错误，非标准JSON: {NAME2PATH_JSON}")

    def _reset_history(self):
        """重置历史文本，处理新的源json文件时调用"""
        self.history_text = ""

    

    def format_single_json(self, single_json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        核心方法：处理单个源json文件的所有数据，按规则生成多条目标格式数据
        :param single_json_data: 单个源json文件的解析结果(dict)
        :return: 该json文件生成的所有目标格式数据列表
        """
        target_data_list = []
        self._reset_history()
        
        # 1. 提取源数据核心公共字段
        src_video_name = single_json_data.get("video", "")
        src_video_start = single_json_data.get("video_start", 0)
        src_video_end = single_json_data.get("video_end", 0)
        src_text_stream = single_json_data.get("text_stream", [])
        src_qa = single_json_data.get("qa", {})
        src_question = src_qa.get("question", "")
        src_cot = src_qa.get("cot", "")
        src_answer = src_qa.get("answer", "")
        src_reasoning_type = src_qa.get("reasoning_type", "")

        # 2. 匹配视频路径：源video是名称 → 目标video是完整路径
        target_video_path = self.name2path.get(src_video_name, "")

        
        # 3. 构建中间thinking数据: 遍历text_stream的每一个item，生成对应目标数据（核心规则：一条item对应一条目标数据）
        for ts_start, ts_end, ts_text in src_text_stream:
            # ========== 核心规则：拼接previous字段（系统prompt + 历史所有文本） ==========
            filled_previous = SYSTEM_PROMPT_TEMPLATE.format(
                prompt=src_question,
                memory=self.history_text
            )
            
            # ========== 构建 user 段数据 ==========
            user_content = [
                {
                    "type": "video",
                    "video": target_video_path,
                    "video_start": float(ts_start),  # 强制转浮点，符合目标格式要求
                    "video_end": float(ts_end)
                },
                {
                    "type": "text",
                    "text": "",
                    "previous": filled_previous,
                    "title": "",
                    "category": None
                }
            ]
            user_dict = {
                "role": "user",
                "content": user_content
            }

            # ========== 构建 assistant 段数据 ==========
            assistant_content = {
                "type": "text_steam",
                "text_stream": [[ts_start, ts_end, ts_text]],
                "qa_stream": []
            }
            assistant_dict = {
                "role": "assistant",
                "content": [assistant_content]
            }


            # ========== 组合单条目标数据并追加 ==========
            single_target_data = [user_dict, assistant_dict]
            target_data_list.append(single_target_data)

            # ========== 核心规则：更新历史文本 → 累加到memory中，供下一个item使用 ==========
            self.history_text += f"Time={ts_start:.1f}s "+ ts_text + "\n"

        # 增加qa片段
        filled_previous = SYSTEM_PROMPT_TEMPLATE.format(
            prompt=src_question,
            memory=self.history_text
        )
        question = TEMPLATE_FINAL_BOXED.format(PromptFinal=src_question)
        
        # ========== 构建 user 段数据 ==========
        user_content = [
            {
                "type": "video",
                "video": target_video_path,
                "video_start": float(ts_start),  # 强制转浮点，符合目标格式要求
                "video_end": float(ts_end)
            },
            {
                "type": "text",
                "text": "",
                "previous": filled_previous,
                "title": "",
                "category": None
            }
        ]
        user_dict = {
            "role": "user",
            "content": user_content
        }

        # ========== 构建 assistant 段数据 ==========
        assistant_content = {
            "type": "text_steam",
            "text_stream": [],
            "qa_stream": [[ts_start, ts_end, question, src_answer]]
        }
        assistant_dict = {
            "role": "assistant",
            "content": [assistant_content]
        }

        # ========== 组合单条目标数据并追加 ==========
        single_target_data = [user_dict, assistant_dict]
        target_data_list.append(single_target_data)


        return target_data_list


    def batch_process_dir(self) -> List[Any]:
        """
        批量处理指定目录下所有json文件，汇总所有生成的目标格式数据
        :return: 全局所有目标数据的大列表
        """
        all_target_data = []
        # 遍历目录下所有文件，只处理.json后缀
        for filename in os.listdir(JSON_SOURCE_DIR):
            if filename.endswith(".json"):
                json_file_path = os.path.join(JSON_SOURCE_DIR, filename)
                # try:
                if True:
                    with open(json_file_path, "r", encoding="utf-8") as f:
                        src_data = json.load(f)
                    # 处理单个json，返回该文件生成的所有目标数据
                    for data in src_data:
                        single_file_target = self.format_single_json(data)
                        all_target_data.extend(single_file_target)
                    print(f"✅ 成功处理: {filename} → 生成 {len(single_file_target)} 条目标数据")
                # except Exception as e:
                #     print(f"❌ 处理失败 {filename}: {str(e)}")
                #     continue
        return all_target_data


def parse_args():
    parser = argparse.ArgumentParser(description="处理数据集路径配置")
    
    # 定义输入目录参数
    parser.add_argument(
        '--json_source_dir', 
        type=str, 
        default='',
        help='输入数据集目录路径'
    )

    parser.add_argument(
        '--name2path_json', 
        type=str, 
        default='',
        help='输入数据集目录路径'
    )

    parser.add_argument(
        '--output_path', 
        type=str, 
        default='',
        help='输入数据集目录路径'
    )
    
    return parser.parse_args()


# ===================== 执行脚本 =====================
if __name__ == "__main__":
    args = parse_args()
    JSON_SOURCE_DIR = args.json_source_dir
    NAME2PATH_JSON = args.name2path_json
    output_path = args.output_path

    # 实例化格式化器
    formatter = JsonDataFormatter()
    # 批量处理所有json
    final_result = formatter.batch_process_dir()
    # 打印最终结果统计
    print(f"\n✅ 全部处理完成！共生成 {len(final_result)} 条目标格式数据")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for single_line_data in final_result:
            json.dump(single_line_data, f, ensure_ascii=False)
            f.write('\n')  # 每行结尾换行，标准jsonl格式
    
    print(f"✅ 结果已保存至: {output_path}")