from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import csv
from pathlib import Path

# ==================== 配置区域 ====================
MODEL_DIR = "Qwen2.5-VL-7B"          # 模型路径
RUMOR_DIR = "weibo/rumor_images"            # 谣言图片文件夹
NONRUMOR_DIR = "weibo/nonrumor_images"      # 非谣言图片文件夹
BATCH_SIZE = 100                            # 每处理多少张图片，就打印一次进度
# ==================================================

# 加载模型（只加载一次）
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)

# 固定的中性描述提示词
prompt = "仅基于图像中可见的客观事实进行中性描述。描述具体物体、人物、场景、文字，不得添加任何主观评价、情感色彩或想象细节。用陈述句客观陈述。"

def get_image_files(folder_path):
    """获取文件夹中所有图片的路径，返回列表"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}
    files = []
    for ext in image_extensions:
        files.extend(Path(folder_path).glob(f"*{ext}"))
        files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
    return sorted(files)  # 按文件名排序，保证顺序一致

def load_existing_names(csv_path):
    """读取CSV中已经处理过的图片名，用于断点续传"""
    if not os.path.exists(csv_path):
        return set()
    names = set()
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader, None)  # 跳过表头
        for row in reader:
            if row:
                names.add(row[0])
    return names

def describe_single_image(image_path):
    """对单张图片生成客观描述"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    return output_text

def process_folder(image_dir, csv_path):
    """处理整个文件夹：遍历图片，生成描述，保存CSV，支持断点续传"""
    # 获取所有图片文件
    all_images = get_image_files(image_dir)
    print(f"文件夹 {image_dir} 中共发现 {len(all_images)} 张图片")

    # 断点续传：读取已处理的图片名
    existing_names = load_existing_names(csv_path)
    print(f"已处理 {len(existing_names)} 张，将跳过它们")

    # 打开CSV文件（追加模式）
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        # 如果是新文件，先写表头
        if not file_exists or os.path.getsize(csv_path) == 0:
            writer.writerow(['image_name', 'description'])

        count = 0
        for img_path in all_images:
            img_name = img_path.name
            # 跳过已处理的图片
            if img_name in existing_names:
                continue

            try:
                description = describe_single_image(img_path)
                writer.writerow([img_name, description])
                f.flush()  # 立刻写入磁盘，防止意外中断丢失数据
                count += 1

                # 每处理 BATCH_SIZE 张，打印一次进度
                if count % BATCH_SIZE == 0:
                    print(f"  已新处理 {count} 张，当前：{img_name}")

            except Exception as e:
                print(f"  [错误] 处理 {img_name} 时出错：{e}")
                # 出错也写入一行标记，方便后续排查
                writer.writerow([img_name, f"ERROR: {e}"])
                f.flush()

    print(f"文件夹 {image_dir} 处理完毕，本次新增 {count} 张描述。")

# ==================== 主程序 ====================
if __name__ == "__main__":
    print("开始处理谣言图片...")
    process_folder(RUMOR_DIR, "rumor_images.csv")

    print("\n开始处理非谣言图片...")
    process_folder(NONRUMOR_DIR, "nonrumor_images.csv")

    print("\n全部完成！")