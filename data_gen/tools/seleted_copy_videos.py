import os 
import json 

import shutil

def copy_files(old_dir, old_path, new_dir):
    '''
        input example:
            old_dir: $DATASET_PATH (root of the raw video datasets)
            old_path: omni_vlr/longvila_videos/hDk_2SCVIFY.mp4
            new_dir: videos/copyed_filtered_videos/
    '''
    try:
        # 1. 拼接完整的源文件路径
        # os.path.join 会自动处理路径分隔符
        src_full_path = os.path.join(old_dir, old_path)

        # 2. 拼接完整的目标文件路径
        # 注意：这里我们保留了 old_path 的目录结构，防止文件名冲突
        # 如果 old_path 开头有 '/'，需要去掉，否则 os.path.join 会忽略 new_dir
        relative_path = old_path.lstrip('/') 
        dst_full_path = os.path.join(new_dir, relative_path)

        # 3. 检查源文件是否存在
        if not os.path.exists(src_full_path):
            print(f"[Warning] Source file not found: {src_full_path}")
            return False

        # 4. 创建目标文件夹（如果不存在）
        # os.path.dirname 获取文件所在的目录路径
        dst_folder = os.path.dirname(dst_full_path)
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder, exist_ok=True)

        # 5. 执行复制
        # shutil.copy2 会保留文件的元数据（如修改时间、权限等），比 copy 更适合数据迁移
        shutil.copy2(src_full_path, dst_full_path)
        
        print(f"[Success] Copied to: {dst_full_path}")
        return True

    except Exception as e:
        print(f"[Error] Failed to copy {old_path}: {e}")
        return False

if __name__ == "__main__":
    old_dir = os.environ.get("DATASET_PATH", "")
    new_dir = "videos/copyed_filtered_videos/"

    json_path = "data_2512/key_result/filted_video.json"
    videos = json.load(open(json_path, 'r'))
    for vid in videos:
        copy_files(old_dir, vid['video_path'], new_dir)
