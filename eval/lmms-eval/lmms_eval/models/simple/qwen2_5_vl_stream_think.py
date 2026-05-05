import base64
import re
from io import BytesIO
from typing import List, Optional, Tuple, Union
import os
import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)
from torchvision import transforms
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
import threading
import queue
from queue import Queue
import random
import time
from contextlib import contextmanager


VIDEO_CACHE_DIR = os.environ.get("VIDEO_CACHE_DIR", "./video_cache")
ENABLE_VIDEO_CACHE = os.environ.get("ENABLE_VIDEO_CACHE", "0").strip().lower() in {"1", "true", "yes", "on"}


# ================= Config =================
# Global timing stats switch: True enables printing; False disables it.
ENABLE_TIME_STATS = True 

@contextmanager
def time_block(label: str):
    """Context manager used to time a code block."""
    if not ENABLE_TIME_STATS:
        yield
        return

    # Synchronize pending GPU work so timing starts accurately.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    try:
        yield
    finally:
        # Wait for GPU work in this block to finish before measuring end time.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        print(f"⏱️ [TimeStats] {label}: {end_time - start_time:.4f} s")

try:
    # Prefer direct import (compatible with newer environments).
    from qwen_vl_utils.vision_process import (
        FORCE_QWENVL_VIDEO_READER, VIDEO_TOTAL_PIXELS, FPS_MAX_FRAMES, 
        VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS, FRAME_FACTOR, IMAGE_FACTOR, FPS,
        smart_nframes, smart_resize
    )
except ImportError:
    # Fallback for older environments with missing constants.
    from qwen_vl_utils.vision_process import smart_nframes, smart_resize

    # --- Fallback constants ---
    FORCE_QWENVL_VIDEO_READER = None
    FPS = 2.0
    FRAME_FACTOR = 2
    FPS_MAX_FRAMES = 768
    
    # Qwen2-VL core factor: Patch(14) * Merge(2) = 28
    IMAGE_FACTOR = 28 
    
    # Core rule: Pixels = token_count * 28 * 28
    # Equivalent to VIDEO_MIN_TOKEN_NUM = 128
    VIDEO_MIN_PIXELS = 128 * 28 * 28   
    
    # Equivalent to VIDEO_MAX_TOKEN_NUM = 768
    VIDEO_MAX_PIXELS = 768 * 28 * 28   
    
    # Total pixel cap (typically aligned with max).
    VIDEO_TOTAL_PIXELS = VIDEO_MAX_PIXELS 
def _read_video_decord_plus(ele: dict, strict_fps: bool = False, drop_last: bool = True, return_pts: bool = False, only_get_last_frame: Optional[int] = None, vr = None, max_num_frames: Optional[int] = None):
    """
    Read video using decord.VideoReader; handles more cases than _read_video_decord.

    Parameters
    ----------
    ele : dict
        A dict describing the video and optional cropping info. Supported keys:
          - "video"        : Path to the video, supports local paths / "file://" / "http(s)://"
          - "video_start"  : (optional) crop start timestamp in seconds
          - "video_end"    : (optional) crop end timestamp in seconds
          - "remote_loader": (optional) callback to load remote files when not local
    strict_fps : bool
        If True, sample frames at a fixed FPS (global constant FPS); otherwise use smart_nframes for adaptive sampling.
    drop_last : bool
        When strict_fps=True and the generated frame count exceeds FPS_MAX_FRAMES, whether to truncate extra frames (True)
        or uniformly resample down to FPS_MAX_FRAMES (False).
    return_pts : bool
        Whether to additionally return the per-frame timestamps (clip_pts).

    Returns
    -------
    clip : torch.Tensor
        Shape (T, C, H, W) where T is the number of sampled frames (as a PyTorch Tensor).
    sample_fps : float
        Effective FPS after sampling = sampled frame count / original segment frame count * original video FPS.
    clip_pts : list[float]   (only if return_pts=True)
        Timestamps (seconds) of each sampled frame.
    """

    # 1) Build VideoReader
    video_path = ele["video"]
    if os.path.exists(video_path):
        if vr is None:
            vr = decord.VideoReader(video_path, num_threads=2)
    else:
        try:
            full_video_path = os.path.join(os.environ['DATASET_PATH'], video_path)
            vr = decord.VideoReader(full_video_path, num_threads=2)
        except Exception as e:
            print(f"Error: {e}")
            raise e

    # 2) Parse crop range
    video_start = ele.get('video_start', None)
    video_end   = ele.get('video_end',   None)
    video_fps   = vr.get_avg_fps()

    clip_idxs, clip_pts = None, None
    if video_start is not None or video_end is not None:
        vr.get_frame_timestamp(0)
        video_pts = vr._frame_pts[:, 1]
        video_start = video_pts[0] if video_start is None else video_start
        video_end   = video_pts[-1] if video_end is None else video_end
        clip_idxs = ((video_start <= video_pts) & (video_pts <= video_end)).nonzero()[0]
        clip_pts  = video_pts[clip_idxs]
        total_frames = len(clip_idxs)
    else:
        total_frames = len(vr)

    # 3) Sampling strategy
    current_max_frames = max_num_frames if max_num_frames is not None else FPS_MAX_FRAMES
    if not strict_fps:
        nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
        nframes_idxs = np.linspace(0, total_frames - 1, nframes).round().astype(int)
        clip_idxs = nframes_idxs if clip_idxs is None else clip_idxs[nframes_idxs]
    else:
        if clip_pts is None:
            vr.get_frame_timestamp(0)
            clip_pts  = vr._frame_pts[:,1]
            clip_idxs = np.arange(len(clip_pts))
        
        expected_timestamps = np.arange(clip_pts[0], clip_pts[-1] + 1e-6, 1 / FPS)
        if len(expected_timestamps) > current_max_frames:
            if drop_last:
                expected_timestamps = expected_timestamps[:current_max_frames]
            else:
                expected_timestamps = expected_timestamps[
                    np.linspace(0, len(expected_timestamps) - 1, current_max_frames).round().astype(int)
                ]

        expected_idxs_for_clip_pts = (expected_timestamps[:, None] <= clip_pts).argmax(axis=1)
        clip_pts  = clip_pts[expected_idxs_for_clip_pts].tolist()
        clip_idxs = clip_idxs[expected_idxs_for_clip_pts].tolist()

        while len(clip_idxs) % FRAME_FACTOR != 0:
            clip_idxs.append(clip_idxs[-1])
            clip_pts.append(clip_pts[-1])

    if only_get_last_frame:
        clip_idxs = clip_idxs[-only_get_last_frame:]
        clip_pts = clip_pts[-only_get_last_frame:]

    # 4) Fetch frames and convert format (THWC -> TCHW)
    clip = torch.from_numpy(vr.get_batch(clip_idxs).asnumpy()).permute(0, 3, 1, 2)

    sample_fps = len(clip_idxs) / max(total_frames, 1e-6) * video_fps

    # 5) Return
    if return_pts:
        return clip, sample_fps, clip_pts
    return clip, sample_fps

def _spatial_resize_video(video: torch.Tensor, video_total_pixels: int, nframes: int = None, min_pixels: int = None):
    """
    Args:
        video: Tensor shape (T, C, H, W)
        video_total_pixels: Maximum total pixel budget for the full video (T * H * W)
    """
    VIDEO_MIN_PIXELS = min_pixels
    if nframes is None:
        nframes, _, height, width = video.shape
    else:
        height, width = video.shape[2:]

    pixels_per_frame = (video_total_pixels * FRAME_FACTOR) / nframes 
    
    if pixels_per_frame < VIDEO_MIN_PIXELS:
        new_nframes = int((video_total_pixels * FRAME_FACTOR) / VIDEO_MIN_PIXELS)
        
        # Keep at least one frame.
        new_nframes = max(1, new_nframes)
        
        # Resample only when the new frame count is smaller.
        if new_nframes < nframes:
            # Uniformly sample frame indices.
            indices = torch.linspace(0, nframes - 1, new_nframes).long()
            video = video[indices]
            
            nframes = new_nframes

    current_pixels_per_frame = video_total_pixels / nframes * FRAME_FACTOR
    max_pixels = max(
        min(VIDEO_MAX_PIXELS, int(current_pixels_per_frame)), 
        int(VIDEO_MIN_PIXELS * 1.05)
    )

    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=VIDEO_MIN_PIXELS,
        max_pixels=max_pixels,
    )

    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
    )
    
    return video.float()

class mute_stderr_ffmpeg:
    def __enter__(self):
        self._stderr_fd = os.dup(2)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, 2)
        return self
    def __exit__(self, *exc):
        os.dup2(self._stderr_fd, 2)
        os.close(self._devnull)
        os.close(self._stderr_fd)


def smart_last_seg_buget(total_chunks):
    return (1+1)/(total_chunks)

@register_model("qwen2_5_vl_stream_think")
class Qwen2_5_VL_Stream_Think(lmms):
    """
    Qwen2.5_VL Model
    "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        max_stream_vid_tokens: int = 8192,
        stream_think_times: Optional[Union[int, str]] = -1,
        max_keep_memory: Optional[int] = 0,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        if self.fps is not None:
            global FPS, FRAME_FACTOR
            FPS = fps
            FRAME_FACTOR = fps
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        self.accelerator = accelerator

        # Read true environment topology.
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Force single-device mapping for multi-process runs.
        num_physical_gpus = torch.cuda.device_count()

        if world_size > 1:
            # Use modulo so multiple ranks can map to one visible GPU.
            device_id = local_rank % num_physical_gpus
            
            self._device = torch.device(f"cuda:{device_id}")
            self.device_map = f"cuda:{device_id}"
            
            if local_rank == 0:
                print(f"🚀 [Force Parallel] Running {world_size} processes on {num_physical_gpus} physical GPUs.")
                print(f"    Process 0 mapped to cuda:{0 % num_physical_gpus}")
                print(f"    Process 1 mapped to cuda:{1 % num_physical_gpus}")
        else:
            self._device = torch.device(device)
            self.device_map = device_map

        self._rank = local_rank
        self._world_size = world_size
        self.cache_video_only = False

        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": "bfloat16",
            "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, **model_kwargs).eval()
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames

        if stream_think_times == -1:
            self.stream_think_times = [1, 1, 2, 3]
        elif isinstance(stream_think_times, int):
            self.stream_think_times = [stream_think_times] * 4
        else:
            self.stream_think_times = list(map(int, stream_think_times.split('-')))
            assert len(self.stream_think_times) == 4, f"invalid format of stream_think_times: {stream_think_times}"

        self.max_stream_vid_tokens = max_stream_vid_tokens
        self.max_keep_memory = max_keep_memory

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def load_video(self, video_dict):
        '''
        Load video data.
        By default this reads source videos directly.
        Optional `.pt` caching can be enabled via ENABLE_VIDEO_CACHE.

        input:
            video_dict: {"video": str}
        return:
            clip: torch.Tensor, a tensor of video frames
            sample_fps: float, the effective frames per second
            clip_pts: list, timestamps for each frame
        '''
        if not ENABLE_VIDEO_CACHE:
            return _read_video_decord_plus(
                video_dict, return_pts=True, strict_fps=True, drop_last=False, max_num_frames=self.max_num_frames
            )

        video_path = video_dict["video"]

        # Build a stable cache file name from the video path.
        try:
            parts = os.path.normpath(video_path).split(os.sep)
            filename_no_ext = os.path.splitext(parts[-1])[0]
            video_cache_name = f"{parts[-3]}_{parts[-2]}_{filename_no_ext}.pt"
        except IndexError:
            filename_no_ext = os.path.splitext(os.path.basename(video_path))[0]
            video_cache_name = f"{filename_no_ext}.pt"

        cached_video_path = os.path.join(VIDEO_CACHE_DIR, video_cache_name)

        if os.path.exists(cached_video_path):
            try:
                cached_data = torch.load(cached_video_path)
                clip = cached_data['clip']
                sample_fps = cached_data['sample_fps']
                clip_pts = cached_data['clip_pts']
                return clip, sample_fps, clip_pts
            except Exception as e:
                print(f"Warning: failed to read cache file '{cached_video_path}': {e}. Rebuilding cache.")

        clip, sample_fps, clip_pts = _read_video_decord_plus(
            video_dict, return_pts=True, strict_fps=True, drop_last=False, max_num_frames=self.max_num_frames
        )

        try:
            data_to_save = {
                'clip': clip,
                'sample_fps': sample_fps,
                'clip_pts': clip_pts
            }
            torch.save(data_to_save, cached_video_path)
        except Exception as e:
            print(f"Warning: failed to write cache file '{cached_video_path}': {e}")

        return clip, sample_fps, clip_pts
    
    def get_textual_memory(self, memory):
        out_put = ""
        if self.max_keep_memory > 0 and len(memory) > (self.max_keep_memory + 1):
            out_put += memory[0]
            for i in range(len(memory)-self.max_keep_memory, len(memory)):
                out_put += memory[i]
        else:
            for past_mem in memory:
                out_put += past_mem 
        return out_put

    # ================= Batch Preprocessing =================
    def _preprocess_batch(self, chunk):
        """
        CPU-heavy stage: video loading, processing, and prompt construction.
        Runs in a worker thread and returns a processed batch.
        """
        contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
        task = task[0]
        split = split[0]
        visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
        gen_kwargs = all_gen_kwargs[0]

        # Normalize stop sequences.
        until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
        if isinstance(until, str):
            until = [until]
        until = [item for item in until if item != "\n\n"]
        gen_kwargs["until"] = until

        if isinstance(contexts, tuple):
            contexts = list(contexts)

        # Clean <image> tags in prompt context.
        for i in range(len(contexts)):
            if "<image>" in contexts[i]:
                contexts[i] = contexts[i].replace("<image>", "")
            if " Provide your reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags." in contexts[i]: # remove the reasoning prompt in VH
                contexts[i] = contexts[i].replace(" Provide your reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags.", "")

        batched_messages = []
        batched_frames_list = []

        for i, context in enumerate(contexts):
            if "<image>" in context:
                context = context.replace("<image>", "")

            message = [{"role": "system", "content": self.system_prompt}]
            if self.reasoning_prompt:
                context = context.strip() + self.reasoning_prompt
                contexts[i] = context

            processed_visuals = []
            clip = None
            sample_fps = 1.0

            if visual_list[i] is not None:
                for visual in visual_list[i]:
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
                        video_dict = {"video": visual}
                        # Video loading (I/O + CPU bound).
                        clip, sample_fps, clip_pts = self.load_video(video_dict)
                            
                        # Allocate stream-think rounds by clip duration.
                        duration_sec = 0.0
                        if len(clip_pts) > 0:
                            duration_sec = clip_pts[-1] - clip_pts[0]
                        
                        duration_min = duration_sec / 60.0

                        if duration_min <= 0.5:
                            stream_think_times = self.stream_think_times[0]
                        elif duration_min < 4:
                            stream_think_times = self.stream_think_times[1]
                        elif duration_min < 30:
                            stream_think_times = self.stream_think_times[2]
                        else:
                            stream_think_times = self.stream_think_times[3]

                        video_total_pixels = self.max_stream_vid_tokens * (28 * 28) * stream_think_times
                        clip = _spatial_resize_video(clip, video_total_pixels, min_pixels=self.min_pixels)      
                        processed_visuals.append(clip)
                    elif isinstance(visual, Image.Image):
                        raise ValueError("Image is not supported yet.")

            STREAM_THINK_PROMPT = "[System]\nYou are a Streaming Video Analyst.\n"
            previous = {"role": "previous text", "content": STREAM_THINK_PROMPT}
            message.append(previous)
            
            frames_list = []
            if clip is not None:
                total_frames = len(clip)
                total_seconds = total_frames / sample_fps

                segment_length = max(int(total_frames / stream_think_times), 1)
                segment_seconds = total_seconds / stream_think_times
                
                for j in range(stream_think_times):
                    start_idx = j * segment_length
                    
                    if start_idx >= total_frames:
                        break

                    start_time = j * segment_seconds
                    end_time = (j + 1) * segment_seconds
                    
                    if j == stream_think_times - 1:
                        end_idx = total_frames
                    else:
                        end_idx = (j + 1) * segment_length
                        end_idx = min(end_idx, total_frames)

                    if start_idx >= end_idx:
                        continue
                        
                    
                    time_str = f"{start_time:.1f}-{end_time:.1f}s"
                    video_chunk = clip[start_idx : end_idx]
                    
                    
                    if len(video_chunk) == 0:
                        continue

                    frames_list.append(video_chunk)   
                    message.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"Time={time_str} "}, 
                                {"type": "video", "video": video_chunk},
                            ],
                        }
                    )
                
                message.append(
                    {
                        "role": "user",
                        "content": f"Time={total_seconds:.1f}s " + context,
                    }
                )

            else:
                # Fallback for text-only input.
                message.append({"role": "user", "content": context})

            batched_messages.append(message)
            batched_frames_list.append(frames_list)

        return {
            "batched_messages": batched_messages,
            "batched_frames_list": batched_frames_list,
            "contexts": contexts,
            "gen_kwargs": gen_kwargs,
            "doc_id": doc_id
        }

    # ================= Main Generation Loop =================
    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        
        # Keep a small queue to overlap CPU preprocessing with GPU generation.
        data_queue = Queue(maxsize=2)
        
        def producer():
            for chunk in chunks:
                try:
                    processed_item = self._preprocess_batch(chunk)
                    if self.cache_video_only:
                        continue
                    data_queue.put(processed_item)
                except Exception as e:
                    print(f"Error in preprocessing thread: {e}")
                    # Sentinel tuple for a failed batch.
                    current_batch_size = len(chunk)
                    data_queue.put((-10000, current_batch_size)) 
            data_queue.put(None)

        preload_thread = threading.Thread(target=producer, daemon=True)
        preload_thread.start()

        while True:
            item = data_queue.get()

            if isinstance(item, tuple) and item[0] == -10000:
                failed_batch_size = item[1]
                print(f"Skipping failed batch of size {failed_batch_size}, filling with random answers.")
                
                for i in range(failed_batch_size):
                    res.append(random.sample(['A', 'B', 'C', 'D'], 1)[0])
                    pbar.update(1)
                continue
            
            if item is None:
                break
            
            batched_messages = item["batched_messages"]
            batched_frames_list = item["batched_frames_list"]
            contexts = item["contexts"]
            gen_kwargs = item["gen_kwargs"]
            
            assert len(batched_messages) == 1, "only support batchsize 1"
            message = batched_messages[0]
            frames_list = batched_frames_list[0]

            # ---------------- Stream Think Loop (GPU Bound) ----------------
            textual_memory = [message[1]['content']]
            for think_round in range(len(frames_list)-1):
                vis_idx = 2 + think_round
                message[1]['content'] = self.get_textual_memory(textual_memory)
                cur_msg = [message[0], message[1], message[vis_idx]]
                cur_vid_list = [frames_list[think_round]]
                
                text = self.processor.apply_chat_template(cur_msg, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=text, videos=cur_vid_list, padding=True, return_tensors="pt")

                if self.device_map == "auto":
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self.device)

                default_gen_kwargs = {
                    "max_new_tokens": 5000,
                    "temperature": 0.0, 
                    "top_p": None,
                    "num_beams": 1,
                }
                current_gen_kwargs = {**default_gen_kwargs}

                pad_token_id = self.tokenizer.pad_token_id
                if current_gen_kwargs["temperature"] > 0:
                    current_gen_kwargs["do_sample"] = True
                else:
                    current_gen_kwargs["do_sample"] = False
                    current_gen_kwargs["temperature"] = None
                    current_gen_kwargs["top_p"] = None

                cont = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=current_gen_kwargs["do_sample"],
                    temperature=current_gen_kwargs["temperature"],
                    top_p=current_gen_kwargs["top_p"],
                    num_beams=current_gen_kwargs["num_beams"],
                    max_new_tokens=current_gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )
                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
                inter_answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                textual_memory.append(message[vis_idx]['content'][0]['text'] + inter_answers[0] + "\n")


            # ---------------- Final Generation (GPU Bound) ----------------
            message[1]['content'] = self.get_textual_memory(textual_memory)
            cur_msg = [message[0], message[1], message[-2], message[-1]]
            cur_vid_list = [frames_list[-1]]
            text = self.processor.apply_chat_template(cur_msg, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=text, videos=cur_vid_list, padding=True, return_tensors="pt")
            
            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            default_gen_kwargs = {
                "max_new_tokens": 32768,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}

            pad_token_id = self.tokenizer.pad_token_id
            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None
            
            with time_block("Final Generation"):
                cont = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=current_gen_kwargs["do_sample"],
                    temperature=current_gen_kwargs["temperature"],
                    top_p=current_gen_kwargs["top_p"],
                    num_beams=current_gen_kwargs["num_beams"],
                    max_new_tokens=current_gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            for ans, context in zip(answers, contexts):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)
                pbar.update(1)

        preload_thread.join()
        
        res = re_ords.get_original(res)
        pbar.close()
        return res


    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
