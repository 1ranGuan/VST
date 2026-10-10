import json
import os
from typing import List, Dict, Any
import glob 

SKIP_TEXT = 'SKIP'

def parse_seconds(time_str: str) -> float:
    """将时间字符串（如 "00:00"）转换为秒数"""
    minutes, seconds = map(int, time_str.split(':'))
    return minutes * 60 + seconds

def wrap_cot_text(cot_text: str) -> str:
    """将cot文本包裹在<cot></cot>标签中"""
    return f"<cot>{cot_text}</cot>"

def wrap_answer_text(answer: str) -> str:
    """将答案文本包裹在<answer></answer>标签中"""
    return f"<answer>{answer}</answer>" 

def cut_unused_segments(cot_segments: List[List[float]]) -> None:
    """裁剪cot_segments中最后持续为SKIP的未使用的片段"""
    if cot_segments[-1][2] != wrap_cot_text(SKIP_TEXT):
        return cot_segments 

    for i in range(len(cot_segments) - 1, 0, -1):
        if cot_segments[i][2] == wrap_cot_text(SKIP_TEXT):
            cot_segments.pop(i)
        else:
            break 
    return cot_segments

def process_json_item(json_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """处理单个JSON条目，转换为user和assistant两个角色的JSON对象"""
    # 提取视频基础信息
    video_name = json_item["VIDEO_NAME"]
    task = json_item["TASK"]
    question = json_item["QUESTION"]
    options = json_item["OPTIONS"]
    answer = json_item["ANSWER"]
    streaming_cot = json_item["STREAMING_COT"]

    # 计算视频总时长（取最后一个片段的结束时间）
    last_clip = streaming_cot[-1] if streaming_cot else {}
    end_time_str = last_clip.get("video_time", ["00:00", "00:00"])[1]
    video_end = parse_seconds(end_time_str)

    cot_segments = []
    for idx, cot_item in enumerate(streaming_cot):
        cot_text = cot_item.get("cot", "").strip()
        cot_text = wrap_cot_text(cot_text)

        start_time_str, end_time_str = cot_item.get("video_time", ["00:00", "00:00"])
        cot_segments.append([parse_seconds(start_time_str), parse_seconds(end_time_str), cot_text])
    
    #去除冗余的<SKIP>片段
    cut_unused_segments(cot_segments)

    # 将cot_segments中的最后一个元素的文本替换为answer
    cot_segments[-1][2] += wrap_answer_text(answer)
    
    assistant_content = [
        {
            "type": "text_stream",
            "text_stream": cot_segments
        }
    ]

    # 构建user角色内容
    options_text = "\n".join([f"{k}) {v}" for k, v in options.items()])
    user_content = [
        {
            "type": "video",
            "video": f"{video_name}.mp4",
            "video_start": 0.0,
            "video_end": round(cot_segments[-1][1], 1)
        },
        {
            "type": "text",
            "text": f"{question} Options: {options_text}",
            "previous": """You are an assistant specializing in video content understanding. You will be provided with a video, along with a multiple-choice question and corresponding options presented at the beginning. As the video plays over time, you need to reason step by step in line with the video content select the optimal timing to output your final answer (containing only the option letter), wrap your entire thinking process with <cot></cot>, and enclose your final answer with<answer></answer>""",
            "title": "",
            "category": task
        }
    ]

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]


def process_json_itemv2(json_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """处理单个JSON条目，转换为user和assistant两个角色的JSON对象"""
    # 提取视频基础信息
    video_name = json_item["videos"]
    problem_id = json_item["problem_id"]
    question = json_item["problem"]
    task = json_item["problem_type"]
    options = json_item["options"]
    answer = json_item["answer"]
    streaming_cot = json_item["reasoning"]

    cot_segments = []
    for idx, cot_item in enumerate(streaming_cot):
        cot_text = cot_item.get("cot", "").strip()
        cot_text = wrap_cot_text(cot_text)

        start_time_str, end_time_str = cot_item.get("video_time", ["00:00", "00:00"])
        cot_segments.append([parse_seconds(start_time_str), parse_seconds(end_time_str), cot_text])

    #去除冗余的<SKIP>片段
    cut_unused_segments(cot_segments)

    # 将cot_segments中的最后一个元素的文本替换为answer
    cot_segments[-1][2] += wrap_answer_text(answer)
    
    assistant_content = [
        {
            "type": "text_stream",
            "text_stream": cot_segments
        }
    ]

    # 构建user角色内容
    # options_text = "\n".join([f"{k}) {v}" for k, v in options.items()])
    user_content = [
        {
            "type": "video",
            "video": f"{video_name}",
            "video_start": 0.0,
            "video_end": round(cot_segments[-1][1], 1)
        },
        {
            "type": "text",
            "text": f"{question} Options: {options}",
            "previous": """You are an assistant specializing in video content understanding. You will be provided with a video, along with a multiple-choice question and corresponding options presented at the beginning. As the video plays over time, if the current video clip does not require the output of the thinking process, directly output SKIP; otherwise, you need to reason step by step in line with the video content, select the optimal timing to output your final answer (containing only the option letter), wrap your entire thinking process with <cot></cot>, and enclose your final answer with<answer></answer>.""",
            "title": "",
            "category": task
        }
    ]

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]


def json_to_jsonl(input_path: str, output_path: str) -> None:
    """
    :param input_path: 输入JSON文件路径（包含数组）
    :param output_path: 输出JSONL文件路径
    """
    # 读取输入JSON
    json_data = []
    json_path_list = glob.glob(input_path+'*.json')
    for json_path in json_path_list:
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                json_data.append(data)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON解析错误: {str(e)}")

    # 处理并写入输出JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in json_data:
            try:
                processed = process_json_itemv2(item)
                # 确保中文正常显示
                json.dump(processed, f, ensure_ascii=False)
                f.write('\n')
            except Exception as e:
                print(f"处理条目失败: {str(e)}, 条目内容: {item}")


if __name__ == "__main__":
    # 示例用法
    INPUT_JSON = "../final_training_data/"    # 输入JSON文件路径
    OUTPUT_JSONL = "../data/curated_sft_streamvlm_format/streamthink_200qa_1124_with_seeks.jsonl"  # 输出JSONL文件路径

    # 检查输入文件是否存在
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"输入文件不存在: {INPUT_JSON}")

    # 执行转换
    json_to_jsonl(INPUT_JSON, OUTPUT_JSONL)
    print(f"转换完成，输出文件: {OUTPUT_JSONL}")