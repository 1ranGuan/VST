"""生成 {video_name: 数据集相对路径} 索引, 供 collect_format.py / format_*.py 把切片目录名映射回原视频路径.
命名规则与 split_clips.py 保持一致: video_name = basename(video).split('.')[0]"""
import argparse
import json
import os


def generate_index(anno_json, output, max_length=2048):
    all_data = []
    for p in anno_json:
        with open(p, 'r', encoding='utf-8') as f:
            all_data += json.load(f)
    print(f"📊 原始数据总条数: {len(all_data)}")

    index_map = {}
    collision_count = 0
    for item in all_data:
        # 过滤条件必须与 split_clips.py 一致, 否则索引会多出未切片的视频
        d_str = str(item.get('duration', 0)).replace('s', '')
        if not d_str:
            continue
        try:
            duration = float(d_str)
        except ValueError:
            continue
        if duration > max_length:
            continue

        rel_path = item.get('video')
        if not rel_path:
            continue
        video_name = os.path.basename(rel_path).split('.')[0]
        if video_name in index_map:
            if index_map[video_name] != rel_path:
                print(f"⚠️ [冲突警告] '{video_name}': {index_map[video_name]} vs {rel_path}")
                collision_count += 1
        else:
            index_map[video_name] = rel_path

    print(f"✅ 有效唯一视频数: {len(index_map)}, 命名冲突: {collision_count}")
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(index_map, f, ensure_ascii=False, indent=4)
    print(f"💾 索引文件已保存至: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_json", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_length", type=float, default=2048)
    args = parser.parse_args()
    generate_index(args.anno_json, args.output, args.max_length)
