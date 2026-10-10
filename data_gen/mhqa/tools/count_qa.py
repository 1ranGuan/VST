import json 
import glob 
from collections import Counter

def count_qa(path):
    json_list = glob.glob(path+"*.json")
    print('Video num: %d'%len(json_list))

    reasoning_type = []
    count = 0
    for p in json_list:
        data = json.load(open(p, 'r'))
        count += len(data)
        for item in data:
            reasoning_type.append(item['qa']['reasoning_type'])
    print('QA num: %d'%count)

    counts = Counter(reasoning_type)
    print(counts)



if __name__ == "__main__":
    p = "mhqa/data/qa_dataset_v0107_filtered_mid_thinking_step3/"
    count_qa(p)

    # p = "mhqa/data/qa_dataset_v0107_filtered_step2/"
    # count_qa(p)

    # p = "mhqa/data/qa_dataset_v0105_step1/"
    # count_qa(p)

    # p = "mhqa/data/qa_dataset_v0105_filtered_step2/"
    # count_qa(p)