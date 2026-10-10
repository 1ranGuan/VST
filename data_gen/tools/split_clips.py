import json
import os
from multiprocessing import Pool
from moviepy.editor import VideoFileClip

# ================= ⚙️ 控制参数 =================
# 并行核数 (根据你的机器配置)
NUM_CORES = 12

# 限制处理的视频数量 (测试用)
# 设置为 10 表示只处理前10个。设置为 None 表示处理所有视频。
# 一共 18k
MAX_PROCESS_NUM = None

# 3级切片阈值
LIMIT_SMALL = 256    # < 256s -> 8段 最大长度 32 s
LIMIT_MEDIUM = 1024  # < 1024s -> 16段 16-64s
MAX_LENGTH = 2048    # < 2048s -> 32段 > 2048s -> 丢弃 32 - 64s

# 路径设置 (命令行参数覆盖)
# --anno_json: 一个或多个 QA 标注 json (每条含 video / duration / id 字段, 如 VideoMarathon 的 multi_choice + open_end)
# --video_root: 原始视频根目录 (标注里的 video 字段相对它)
# --output_root: 切片输出目录, 每个视频一个子目录: <output_root>/<video_name>/part_{i}.mp4 + ids.json
# ==============================================

def process_one_video(args):
    """
    args: (rel_path, video_info, v_root, out_root)
    video_info 是一个字典: {'duration': 123.4, 'ids': ['id_1', 'id_2'...]}
    """
    rel_path, info, v_root, out_root = args
    
    duration = info['duration']
    related_ids = info['ids'] # 获取该视频关联的所有ID
    
    # 1. 确定切片数
    if duration < LIMIT_SMALL:
        num_slices = 8
    elif duration < LIMIT_MEDIUM:
        num_slices = 16
    else:
        num_slices = 32

    # 2. 准备路径
    full_path = os.path.join(v_root, rel_path)
    video_name = os.path.basename(rel_path).split('.')[0]
    save_dir = os.path.join(out_root, video_name)
    
    # 创建文件夹
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # --- 新增: 保存 ID 索引文件 ---
    # 无论视频是否切片过，都更新一下 ID 文件，确保索引是最新的
    id_file_path = os.path.join(save_dir, "ids.json")
    with open(id_file_path, 'w', encoding='utf-8') as f:
        json.dump(related_ids, f, indent=2)
    # ---------------------------

    # 3. 检查是否已切片完成 (断点续传)
    if len([n for n in os.listdir(save_dir) if n.endswith('.mp4')]) >= num_slices:
        print(f"[跳过] {video_name} 视频切片已存在 (ID已保存)")
        return

    print(f"▶️ 处理: {video_name} (关联 {len(related_ids)} 条数据)")

    # 4. 切片逻辑
    try:
        clip = VideoFileClip(full_path)
        real_duration = clip.duration
        part_duration = real_duration / num_slices

        for i in range(num_slices):
            start = i * part_duration
            end = (i + 1) * part_duration

            sub = clip.subclip(start, end)
            out_name = os.path.join(save_dir, f"part_{i}.mp4")
            
            sub.write_videofile(out_name, codec="libx264", audio=False, preset='ultrafast', logger=None)

        clip.close()
    except Exception as e:
        print(f"❌ 错误 {video_name}: {e}")


if __name__ == "__main__":
    # 1. 读取并合并数据
    import argparse
    parser = argparse.ArgumentParser(description="按时长把视频等分成 8/16/32 段 part_{i}.mp4")
    parser.add_argument("--anno_json", nargs="+", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--num_cores", type=int, default=NUM_CORES)
    parser.add_argument("--max_process_num", type=int, default=MAX_PROCESS_NUM)
    args = parser.parse_args()
    video_root, output_root = args.video_root, args.output_root
    NUM_CORES, MAX_PROCESS_NUM = args.num_cores, args.max_process_num

    print("正在读取 JSON 数据...")
    all_data = []
    for p in args.anno_json:
        with open(p, 'r', encoding='utf-8') as f:
            all_data += json.load(f)

    # 2. 聚合数据 (按视频路径分组，收集所有 ID)
    valid_videos = {}

    for item in all_data:
        d_str = str(item.get('duration', 0)).replace('s', '')
        if not d_str: continue
        duration = float(d_str)

        if duration > MAX_LENGTH:
            continue

        video_path = item.get('video')
        # 假设每条数据都有一个唯一 ID，键名通常是 'id' 或 'question_id'
        # 请根据你的实际 JSON 修改这里，比如 item.get('id')
        data_id = item.get('id') 

        if video_path:
            if video_path not in valid_videos:
                valid_videos[video_path] = {
                    'duration': duration,
                    'ids': []  # 初始化 ID 列表
                }
            
            # 把当前这条数据的 ID 加入列表
            if data_id:
                valid_videos[video_path]['ids'].append(data_id)

    # 转换成列表
    video_list = list(valid_videos.items())
    print(f"去重后总视频数: {len(video_list)}")

    # 3. 截断测试
    if MAX_PROCESS_NUM is not None:
        video_list = video_list[:MAX_PROCESS_NUM]
        print(f"⚠️ 测试模式: 仅处理前 {len(video_list)} 个")

    # 4. 准备任务
    tasks = []
    for path, info in video_list:
        tasks.append((path, info, video_root, output_root))

    # 5. 启动多进程
    print(f"🚀 启动多进程 (Cores: {NUM_CORES})...")
    with Pool(processes=NUM_CORES) as pool:
        pool.map(process_one_video, tasks)

    print("\n🎉 完成！每个文件夹下现已包含 ids.json")