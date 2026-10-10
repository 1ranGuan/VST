import os
import json
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# def get_video_duration(file_path):
#     """
#     单独提取的函数：获取单个视频的时长
#     """
#     try:
#         cap = cv2.VideoCapture(file_path)
#         if cap.isOpened():
#             fps = cap.get(cv2.CAP_PROP_FPS)
#             frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#             cap.release()
            
#             if fps > 0:
#                 return frame_count / fps
#     except Exception:
#         pass
#     return 0

import subprocess

path = "data_2512/key_result/filted_vm_1224.json"
existing_files = json.load(open(path, 'r', encoding="utf-8"))

def get_video_duration(file_path):
    """使用 ffprobe 获取时长，通常比 cv2 更轻量"""
    cmd = [
        'ffprobe', 
        '-v', 'error', 
        '-show_entries', 'format=duration', 
        '-of', 'default=noprint_wrappers=1:nokey=1', 
        file_path
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return float(output)
    except:
        return 0


def process_single_folder(video_root_path, video_name, min_dur, max_dur):
    """
    处理单个视频文件夹的任务函数
    返回: (video_name, bool) -> (视频名, 是否符合条件)
    """
    video_path = os.path.join(video_root_path, video_name)
    
    # 快速检查路径是否存在
    if not os.path.isdir(video_path):
        return video_name, False

    try:
        # 获取该目录下所有符合条件的文件
        files = [f for f in os.listdir(video_path) if f.endswith(".mp4") and f.startswith("part_")]
        
        video_duration = 0
        for file_name in files:
            full_path = os.path.join(video_path, file_name)
            video_duration += get_video_duration(full_path)
            if video_duration > max_dur:
                return video_name, False
        
        if min_dur <= video_duration <= max_dur:
            return video_name, True
                
    except Exception as e:
        # print(f"Error accessing {video_name}: {e}")
        pass
        
    return video_name, False

def filter_videos_fast(root_dir, output_json_path, min_duration_sec=60, max_duration_sec=2048, workers=32):
    if not os.path.exists(root_dir):
        print(f"错误: 目录 {root_dir} 不存在")
        return

    # 1. 快速列出所有目录
    print("正在扫描目录结构...")
    all_video_names = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    all_video_names = [d for d in all_video_names if d not in existing_files]
    print(f"共发现 {len(all_video_names)} 个视频目录，准备开始并发处理...")
    
    valid_video_names = []
    
    # 2. 使用线程池并发处理
    # 对于挂载盘(KS3/S3)，建议 workers 设置大一点 (例如 32 或 64)，因为主要瓶颈是网络延迟
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # 提交所有任务
        future_to_video = {
            executor.submit(process_single_folder, root_dir, v_name, min_duration_sec, max_duration_sec): v_name 
            for v_name in all_video_names
        }
        
        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(future_to_video), total=len(all_video_names), desc="Filtering"):
            video_name, is_valid = future.result()
            if is_valid:
                valid_video_names.append(video_name)

    # 3. 排序并保存
    valid_video_names.sort()
    
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(valid_video_names, f, indent=4, ensure_ascii=False)
        print(f"\n完成！共筛选出 {len(valid_video_names)} 个符合条件的 video_name")
        print(f"结果已保存至: {output_json_path}")
    except Exception as e:
        print(f"写入 JSON 失败: {e}")

if __name__ == "__main__":
    ROOT_DIR = "videos/vm_video_clip"
    OUTPUT_JSON = "vm_filter_fast.json"
    
    LIMIT_SECONDS = 256 * 8 
    
    # 这里的 workers=32 是关键，针对 KS3 这种网络存储，可以尝试开到 64 甚至更高
    filter_videos_fast(ROOT_DIR, OUTPUT_JSON, min_duration_sec=60, max_duration_sec=LIMIT_SECONDS, workers=64)
