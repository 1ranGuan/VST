from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import torch.multiprocessing as mp
import json
import os
import os.path as osp
from tqdm import tqdm
import math
from datetime import datetime
import re
import logging
import time
from collections import defaultdict
import argparse
import ffmpeg
import sys
import shutil
from decord import VideoReader, cpu
from PIL import Image

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

# Parameters
RUN_NAME = ""
CKPT_PATH = ""

TASK_JSON = ""
VIDEO_DIR = ""
RESULT_DIR = "eval/result_streamingbench"
LOG_PATH = "log/{run_name}_{curr_time}.log"
OUTPUT_JSONL = "output/{run_name}_{curr_time}.jsonl"
MIN_PIXELS = 336*336
MAX_PIXELS = 448*448
MIN_FRAMES = 4
MAX_FRAMES = 56

# Prompt template
prompt = """You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

Question: {}

Options:
{}

The best option is:"""

# helper functions
def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")
    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]

def split_video(video_file, start_time, end_time, logger):
    """
    Split video into prefix part based on timestamp.
    """
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_dir = os.path.join(os.path.dirname(video_file), "tmp_60")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{video_name}_{start_time}_{end_time}.mp4")
    
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        return output_file
        
    try:
        temp_output = output_file + f".part_{os.getpid()}"
        (
            ffmpeg
            .input(video_file, ss=float(start_time))
            .output(temp_output, t=(float(end_time) - float(start_time)), vcodec='libx264', acodec='aac', loglevel="quiet")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        os.rename(temp_output, output_file)
    except ffmpeg.Error as e:
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            return output_file
        logger.error(f"ffmpeg error: {e.stderr.decode('utf-8') if e.stderr else str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error splitting video: {e}")
        return None
    return output_file

def run_inference(rank, world_size, args, shared_data):
    curr_time = shared_data['curr_time']
    
    sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
    from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    
    # Set up logging for this rank
    log_path_rank = osp.join(args.result_dir, f"log/{args.run_name}_{curr_time}_rank{rank}.log")
    logger = logging.getLogger(f"worker_{rank}")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)7s | [GPU{}] %(message)s".format(rank))
    fh = logging.FileHandler(log_path_rank)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # Set device
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Load task info from JSON and split
    with open(args.task_json, 'r', encoding='utf-8') as f:
        task_data = json.load(f)
    
    chunk_size = math.ceil(len(task_data) / world_size)
    my_data = task_data[rank * chunk_size : (rank + 1) * chunk_size]
    
    output_jsonl_rank = osp.join(args.result_dir, f"output/{args.run_name}_{curr_time}_rank{rank}.jsonl")
    
    logger.info(f"Worker {rank} processing {len(my_data)} tasks")
    
    # Load model and processor
    torch.manual_seed(1234)
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.ckpt_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": rank},
        )
        
        processor = AutoProcessor.from_pretrained(
            args.ckpt_path,
            min_pixels=args.min_pixels, 
            max_pixels=args.max_pixels * 4, 
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Inference loop
    for item in tqdm(my_data, total=len(my_data), position=rank, desc=f"GPU {rank}"):
        try:
            task_type = item.get('subtask', 'Unknown')
            question = item['question']
            start_time = float(item['start'])
            end_time = float(item['end'])
            options = item['candidates']
            answer_text = item['answer']
            video_rel_path = item['video']

            # Map correct answer text to A, B, C, D
            try:
                answer_idx = options.index(answer_text)
                answer_letter = chr(ord('A') + answer_idx)
            except ValueError:
                logger.warning(f"Answer text not found in candidates for video {video_rel_path}")
                answer_letter = "Unknown"

            # Format options for prompt
            formatted_options = [f"{chr(ord('A')+i)}. {opt}" for i, opt in enumerate(options)]
            
            video_path = osp.join(args.video_dir, video_rel_path)
            
            # Split video
            video_path = split_video(video_path, start_time, end_time, logger)
            if not video_path: continue

            # StreamForest Logic
            content_list = [{"type": "text", "text": f"Time={start_time:.1f}-{end_time:.1f}s "}]
            
            try:
                vr = VideoReader(video_path, ctx=cpu(0))
                total_frames = len(vr)
                video_fps = vr.get_avg_fps()
                
                target_fps = 1.0
                if 300 < end_time <= 600:
                    target_fps = 0.5
                elif end_time > 600:
                    target_fps = 0.2
                
                step = max(1, int(video_fps / target_fps))
                frame_indices = list(range(0, total_frames, step))
                
                if len(frame_indices) > args.max_frames:
                    frame_indices = frame_indices[-args.max_frames:]
                    start_time = end_time - (len(frame_indices) - 1) * step / video_fps
                
                history_frames_pil = []
                last_frame_pil = None
                
                for i, idx in enumerate(frame_indices):
                    frame_np = vr[idx].asnumpy()
                    pil_img = Image.fromarray(frame_np)
                    
                    is_last_frame = (i == len(frame_indices) - 1)
                    w, h = pil_img.size
                    curr_pixels = w * h
                    
                    if is_last_frame:
                        target_pixels = args.max_pixels * 4
                        scale_factor = (target_pixels / curr_pixels) ** 0.5
                        new_w = int(w * scale_factor)
                        new_h = int(h * scale_factor)
                        last_frame_pil = pil_img.resize((new_w, new_h), Image.BICUBIC)
                    else:
                        target_pixels = args.max_pixels 
                        scale_factor = (target_pixels / curr_pixels) ** 0.5
                        if scale_factor < 1.0:
                            new_w = int(w * scale_factor)
                            new_h = int(h * scale_factor)
                            pil_img = pil_img.resize((new_w, new_h), Image.BICUBIC)
                        
                        history_frames_pil.append(pil_img)

                if history_frames_pil:
                    content_list.append({
                        "type": "video",
                        "video": history_frames_pil,
                        "fps": target_fps,
                    })
                
                if last_frame_pil:
                    content_list.append({
                        "type": "image",
                        "image": last_frame_pil,
                    })
                    
            except Exception as e:
                logger.error(f"Error in manual frame processing: {e}")
                content_list.append({
                    "type": "video",
                    "video": video_path,
                    "max_pixels": args.max_pixels,
                    "fps": target_fps 
                })

            content_list.append({
                "type": "text", 
                "text":  f"\nTime={end_time:.1f}s "+prompt.format(question, '\n'.join(formatted_options))
            })

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "previous text", "content": "[System]\nYou are a Streaming Video Analyst.\n"},
                {
                    "role": "user",
                    "content": content_list,
                }
            ]
            
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
            )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = output_text[0]
            
            output_dict = {
                'task_type': task_type,
                'question': question,
                'start_time': start_time,
                'end_time': end_time,
                'answer': answer_letter,
                'options': formatted_options,
                'video': video_rel_path,
                'response': response
            }
            with open(output_jsonl_rank, 'a') as f:
                f.write(json.dumps(output_dict) + '\n')
        except Exception as e:
            logger.error(f"Error in processing item: {e}")

### Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--task_json", type=str, default=TASK_JSON)
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--min_frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_NAME = args.run_name
    CKPT_PATH = args.ckpt_path
    RESULT_DIR = args.result_dir
    TASK_JSON = args.task_json
    VIDEO_DIR = args.video_dir
    LOG_PATH = osp.join(RESULT_DIR, LOG_PATH.format(run_name=RUN_NAME, curr_time=curr_time))
    OUTPUT_JSONL = osp.join(RESULT_DIR, OUTPUT_JSONL.format(run_name=RUN_NAME, curr_time=curr_time))
    MIN_PIXELS = args.min_pixels
    MAX_PIXELS = args.max_pixels
    MIN_FRAMES = args.min_frames
    MAX_FRAMES = args.max_frames
    
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'output'), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'log'), exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)7s | %(message)s")
    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())
    
    logger.info(f"Running {RUN_NAME} on StreamingBench with {args.world_size} GPUs")
    
    shared_data = {'curr_time': curr_time}
    start_time = time.time()
    
    if args.debug:
        logger.warning("Debug mode enabled: Running single process")
        run_inference(0, 1, args, shared_data)
    else:
        mp.set_start_method('spawn', force=True)
        mp.spawn(run_inference, args=(args.world_size, args, shared_data), nprocs=args.world_size, join=True)
        
    end_time = time.time()
    cost_time = int(end_time - start_time)
    
    logger.info("Merging results...")
    with open(OUTPUT_JSONL, 'w') as outfile:
        for rank in range(args.world_size):
            rank_file = osp.join(RESULT_DIR, f"output/{RUN_NAME}_{curr_time}_rank{rank}.jsonl")
            if os.path.exists(rank_file):
                with open(rank_file, 'r') as infile:
                    shutil.copyfileobj(infile, outfile)

    cnt_total = defaultdict(int)
    cnt_correct = defaultdict(int)
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, 'r') as f:
            lines = f.readlines()
        for line in lines:
            item = json.loads(line)
            cnt_total['overall'] += 1
            cnt_total[item['task_type']] += 1
            if extract_characters_regex(item['response']) == item['answer']:
                cnt_correct['overall'] += 1
                cnt_correct[item['task_type']] += 1
    
    task_types = ['Object Perception', 'Causal Reasoning', 'Clips Summarize', 'Attribute Perception', 'Event Understanding', 'Text-Rich Understanding', 'Prospective Reasoning', 'Spatial Understanding', 'Action Perception', 'Counting']
    for task_type in task_types:
        if cnt_total[task_type] == 0:
            logger.info(f"- {task_type}: No question processed")
        else:
            logger.info(f"- {task_type}: {cnt_correct[task_type]}/{cnt_total[task_type]} = {100*cnt_correct[task_type]/cnt_total[task_type]:.2f}%")
    if cnt_total['overall'] == 0:
        logger.info("No question processed")
    else:
        logger.info(f"Total: {cnt_total['overall']}, Correct: {cnt_correct['overall']}, Accuracy: {100*cnt_correct['overall']/cnt_total['overall']:.2f}%")
    
    logger.info(f"Inference cost time: {cost_time // 3600}h {(cost_time % 3600) // 60}m {cost_time % 60}s")
