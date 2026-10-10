import json
import argparse

def get_jsonl_line_byte_counts(jsonl_path: str, encoding: str = 'utf-8') -> list[int]:
    """
    读取 JSONL 文件，计算每行的字节数，返回字节数列表
    
    参数:
        jsonl_path: JSONL 文件路径
        encoding: 文件编码格式，默认 'utf-8'
        
    返回:
        list[int]: 每行的字节数组成的列表
        
    异常:
        FileNotFoundError: 文件不存在时抛出
        IOError: 文件读写错误时抛出
    """
    byte_counts = []
    accumulated_bytes = 0.0
    with open(jsonl_path, 'r', encoding=encoding) as f:
        for line in f:
            # 计算当前行的字节数（需用相同编码encode）
            line_bytes = line.encode(encoding)
            byte_counts.append(int(accumulated_bytes))
            accumulated_bytes += len(line_bytes)
    return byte_counts

def write_seeks(jsonl_path: str, byte_counts: list[int]):
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        json.dump(byte_counts, f, ensure_ascii=False)

# 示例用法
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video streaming data for training and validation.")
    
    # 命令行参数定义
    parser.add_argument(
        '--input_path', 
        type=str, 
        required=True,
        help='Path to the input JSONL file.'
    )
    args = parser.parse_args()

    try:
        # byte_lengths = get_jsonl_line_byte_counts("../data/curated_sft_streamvlm_format/streamthink_2kqa_1124_with_seeks.jsonl")
        # write_seeks("../data/curated_sft_streamvlm_format/streamthink_2kqa_1124_seeks.jsonl", byte_lengths)
        p = args.input_path
        # p = 'data/curated_sft_streamvlm_format/Streaming_cot_vm_60k_1226_valid_with_seeks.jsonl'
        byte_lengths = get_jsonl_line_byte_counts(p)

        write_seeks(p.replace('with_seeks', 'seeks'), byte_lengths)
        # print("每行字节数:", byte_lengths)
        print("总行数:", len(byte_lengths))
    except (FileNotFoundError, IOError) as e:
        print(f"错误: {e}")