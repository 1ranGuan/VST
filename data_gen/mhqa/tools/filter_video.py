import json 
import os 
from tqdm import tqdm

def filter_videos(data, filtered_videos):
    res_data = []
    for item in tqdm(data):
        print(item.keys())
        if os.path.basename(item['video_name']) in filtered_videos:
            res_data.append(item)
    return res_data

if __name__ == "__main__":
    data = json.load(open('mhqa/aggregated_captions_1.json', 'r'))
    filtered_videos = json.load(open('data_2512/key_result/filted_video.json', 'r'))
    filtered_videos = [item['video_path'] for item in filtered_videos]
    new_data = filter_videos(data, filtered_videos)
    print('after filtering: ', len(new_data))
