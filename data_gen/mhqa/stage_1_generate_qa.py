import os
from qaGenerator import *


def process_single_folder(folder_name, run_log_file_path):
    """
    单个视频的处理逻辑
    :param folder_name: 视频文件夹名称 (也是 ID)
    :param run_log_file_path: 运行监控日志路径
    :return: (folder_name, metric_value) -> metric可以是费用或生成数量
    """
    try:
        # 1. 构造路径
        video_root = os.environ.get("VIDEO_PATH", "")
        full_video_path = os.path.join(video_root, folder_name)
        
        # 构造该视频对应的 Graph Log 路径
        # 假设之前的步骤生成的 log 命名规则是: {folder_name}_graph.jsonl
        # 请根据你的实际情况修改这里的路径拼接逻辑
        graph_log_root = os.environ.get("GRAPH_LOG_ROOT", "./graph_logs") 
        graph_log_path = os.path.join(graph_log_root, f"{folder_name}.jsonl") # 或者 .json
        
        output_folder = os.environ.get("OUTPUT_FOLDER", "./qa_dataset")

        # 2. 预检查
        if not os.path.exists(full_video_path):
            # 记录错误到日志
            with open(run_log_file_path, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] [WARN] Video path not found: {full_video_path}\n")
            return None, 0
            
        # 3. 初始化生成器
        generator = VideoGraphQAGenerator(
            video_folder=folder_name, # 注意：这里传相对路径或 ID，内部会拼 VIDEO_PATH
            log_file_path=graph_log_path,
            qa_num=2,
            output_folder=output_folder
        )
        
        # 4. 执行生成
        # 假设 generator.run() 返回生成的 QA 列表
        results = generator.run()
        
        qa_count = len(results)

        print(qa_count)
        
        # 如果生成了数据，视为成功
        if qa_count > 0:
            return folder_name, qa_count
        else:
            # 可能是图太浅被跳过，或者 API 失败
            print('error, len(qa_count)==0')
            return None, 0

    except Exception as e:
        try:
            with open(run_log_file_path, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] [CRITICAL] [{folder_name}] {e}\n")
        except:
            pass
        return None, 0

# ================= 主程序 =================
if __name__ == "__main__":
    # 0. 基础环境检查
    if "VIDEO_PATH" not in os.environ:
        print("❌ Error: os.environ['VIDEO_PATH'] is not set.")
        exit(1)
        
    filted_file = os.environ.get("FILTED_VM_1224")
    if not filted_file or not os.path.exists(filted_file):
        print(f"❌ Error: Filter file not found: {filted_file}")
        # 如果是测试，可以 mock 一个列表
        # all_items = ["video_001", "video_002"]
        exit(1)
    else:
        with open(filted_file, 'r', encoding='utf-8') as f:
            all_items = json.load(f)

    video_folders = random.sample(all_items, min(len(all_items), 10000))
    
    # 1. 加载外部已处理列表 (断点续传)
    already_processed_set = set()
    if EXISTING_PROCESSED_FILE and os.path.exists(EXISTING_PROCESSED_FILE):
        try:
            with open(EXISTING_PROCESSED_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    already_processed_set = set(data)
            print(f"📋 Loaded external processed list: {len(already_processed_set)} items.")
        except Exception as e:
            print(f"⚠️ Failed to load external processed list: {e}")

    # 2. 过滤逻辑
    target_folders = []
    skipped_count = 0
    processed_success_list = list(already_processed_set) # 保持历史记录
    
    # 原始筛选 (Debug Limit)
    candidates = video_folders[:DEBUG_LIMIT] if DEBUG_LIMIT else video_folders
    
    for folder in candidates:
        if folder in already_processed_set:
            skipped_count += 1
        else:
            target_folders.append(folder)

    mode_str = "Debug Mode" if DEBUG_LIMIT else "Full Mode"
    print(f"🚀 {mode_str}: Found {len(candidates)} folders.")
    print(f"⏭️ Skipped {skipped_count} folders (Already Processed).")
    print(f"✅ Actual Targets: {len(target_folders)} folders to process.")

    # 3. 初始化日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 确保日志目录存在
    os.makedirs("process_vm", exist_ok=True)
    
    json_filename = EXISTING_PROCESSED_FILE # 直接更新这个文件，或者用带时间戳的新文件
    log_filename = f"process_vm/qa_gen_monitor_{timestamp}.log"
    
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"=== Start QA Generation at {timestamp} ===\n")
        f.write(f"Total Candidates: {len(candidates)}\n")
        f.write(f"Skipped: {skipped_count}\n")
        f.write(f"Actual Targets: {len(target_folders)}\n")
        f.write(f"Max Workers: {MAX_WORKERS}\n\n")

    print(f"🔥 Starting Pool (Workers: {MAX_WORKERS})...")
    print(f"📜 Monitor Log: {log_filename}")
    print(f"📊 Running...\n")
    
    # 统计生成的 QA 总数 (替代 Cost)
    total_qa_generated = 0

    # 4. 并发执行
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_folder = {
            executor.submit(process_single_folder, folder, log_filename): folder 
            for folder in target_folders
        }
        
        for future in tqdm(as_completed(future_to_folder), total=len(target_folders), desc="QA Gen Progress", unit="vid"):
            folder = future_to_folder[future]
            try:
                # 解包返回值：(folder_name, qa_count)
                result_folder, result_count = future.result()
                
                if result_folder:
                    # 成功生成了 QA
                    if result_folder not in processed_success_list:
                        processed_success_list.append(result_folder)
                    
                    total_qa_generated += result_count
                    
                    tqdm.write(f"✅ [SUCCESS] '{result_folder}': Generated {result_count} QAs | 📈 Total QAs: {total_qa_generated}")
                    
                    # 记录到日志
                    with open(log_filename, "a", encoding="utf-8") as f:
                        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] [OK] {result_folder}: {result_count} QAs\n")

                    # 实时保存已处理列表
                    try:
                        with open(json_filename, "w", encoding="utf-8") as f:
                            json.dump(processed_success_list, f, ensure_ascii=False, indent=4)
                    except Exception:
                        pass
                else:
                    # 返回 None，说明被跳过（图太浅或出错）
                    # tqdm.write(f"⚠️ [SKIP] '{folder}' (Graph too shallow or Error)")
                    pass
                        
            except Exception as exc:
                tqdm.write(f"❌ Exception in {folder}: {exc}")

    # 5. 结束汇总
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(f"\n=== All Done at {datetime.now().strftime('%H:%M:%S')} ===\n")
        f.write(f"Total Processed Videos: {len(processed_success_list)}\n")
        f.write(f"Total QA Pairs Generated: {total_qa_generated}\n")

    print(f"\n✅ All Done!")
    print(f"📊 Total QA Pairs Generated: {total_qa_generated}")
    print(f"📝 Check {log_filename} for detailed logs.")