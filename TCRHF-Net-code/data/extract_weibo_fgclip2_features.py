
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)


# ============================================================
# 允许读取部分损坏但仍可打开的图片
# ============================================================

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# 路径配置
#
# 当前代码保存位置：
# D:\TIC_code\data\extract_weibo_fgclip2_features.py
#
# 数据文件：
# D:\TIC_code\data\weibo\weibo_text.xlsx
# D:\TIC_code\data\weibo\weibo_image_description.xlsx
# D:\TIC_code\data\weibo\weibo_image\
#
# 模型：
# D:\TIC_code\FG-CLIP2-Large\
#
# 输出：
# D:\TIC_code\data\feature\weibo\
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TEXT_EXCEL_PATH = BASE_DIR / "weibo" / "weibo_text.xlsx"
DESC_EXCEL_PATH = BASE_DIR / "weibo" / "weibo_image_description.xlsx"
IMAGE_DIR = BASE_DIR / "weibo" / "weibo_image"

MODEL_DIR = BASE_DIR.parent / "FG-CLIP2-Large"

OUTPUT_DIR = BASE_DIR / "feature" / "weibo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 参数配置
# ============================================================

# FG-CLIP2-Large long-text 模式
TEXT_MAX_LENGTH = 196
TEXT_WALK_TYPE = "long"

# 文本批大小。显存不足时可改为 16 / 8
TEXT_BATCH_SIZE = 32

# 与你提供的 GossipCop 特征提取代码保持一致：
# True  = 保存 L2 归一化后的 1024 维特征
# False = 保存模型原始 1024 维特征
L2_NORMALIZE = True

# False：断点续跑，已存在的 .npy 不重复计算
# True ：覆盖并重新提取全部特征
OVERWRITE = False

FEATURE_DIM = 1024

# 中文本身不受 lower() 影响；夹杂英文时会转小写。
# 与参考代码保持一致。
LOWERCASE_TEXT = True


# ============================================================
# 设备配置
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"检测到 GPU：{torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("未检测到 GPU，将使用 CPU。")


# ============================================================
# 基础检查
# ============================================================

for p, name in [
    (TEXT_EXCEL_PATH, "weibo_text.xlsx"),
    (DESC_EXCEL_PATH, "weibo_image_description.xlsx"),
    (IMAGE_DIR, "weibo_image 文件夹"),
    (MODEL_DIR, "FG-CLIP2-Large 模型文件夹"),
]:
    if not p.exists():
        raise FileNotFoundError(f"找不到 {name}：\n{p}")


# ============================================================
# 加载 FG-CLIP2-Large
# ============================================================

print("=" * 90)
print("正在加载 FG-CLIP2-Large")
print(f"模型路径：{MODEL_DIR}")
print("=" * 90)

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_DIR),
    trust_remote_code=True,
    local_files_only=True,
)
model = model.to(DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR),
    trust_remote_code=True,
    local_files_only=True,
)

image_processor = AutoImageProcessor.from_pretrained(
    str(MODEL_DIR),
    trust_remote_code=True,
    local_files_only=True,
)

print("FG-CLIP2-Large 加载完成。")


# ============================================================
# 工具函数
# ============================================================

def normalize_feature(feature):
    """对最后一维做 L2 归一化。"""
    norm = torch.norm(
        feature,
        p=2,
        dim=-1,
        keepdim=True,
    )
    return feature / torch.clamp(norm, min=1e-12)


def determine_max_num_patches(image):
    """
    沿用参考代码中的动态图片 patch 策略。
    """
    width, height = image.size
    max_val = (width // 16) * (height // 16)

    if max_val > 784:
        return 1024
    elif max_val > 576:
        return 784
    elif max_val > 256:
        return 576
    elif max_val > 128:
        return 256
    else:
        return 128


def move_to_device(inputs):
    """把 tokenizer / image processor 的 Tensor 移到设备。"""
    result = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            result[key] = value.to(DEVICE)
        else:
            result[key] = value
    return result


def clean_text(text):
    """文本清理，与参考代码保持同样逻辑。"""
    if text is None:
        return ""

    text = str(text).strip()

    if text.lower() in {"nan", "none", "null", "na"}:
        return ""

    # 只合并空白，不改变中文字符本身
    text = " ".join(text.split())

    if LOWERCASE_TEXT:
        text = text.lower()

    return text


def save_feature(feature, save_path):
    """保存为 shape=(1024,) 的 float32 npy。"""
    feature = feature.detach().float().cpu().numpy().reshape(-1)

    if feature.shape != (FEATURE_DIM,):
        raise ValueError(
            f"特征维度错误：{save_path.name}，"
            f"得到 {feature.shape}，期望 ({FEATURE_DIM},)"
        )

    np.save(save_path, feature.astype(np.float32))


def save_zero_feature(save_path):
    """缺失模态保存 1024 维零向量，以保证编号不乱。"""
    np.save(
        save_path,
        np.zeros(FEATURE_DIM, dtype=np.float32),
    )


# ============================================================
# 图片路径查找
# ============================================================

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".gif",
    ".tif",
    ".tiff",
]


def find_image(image_name):
    """
    根据 Excel 中 image_name 查找原图。
    支持：
    1. 完整文件名；
    2. 无后缀文件名；
    3. 常见图片后缀；
    4. 大小写后缀差异。
    """
    image_name = str(image_name).strip()

    if not image_name:
        return None

    direct_path = IMAGE_DIR / image_name
    if direct_path.exists() and direct_path.is_file():
        return direct_path

    stem = Path(image_name).stem

    for ext in IMAGE_EXTENSIONS:
        candidate = IMAGE_DIR / f"{stem}{ext}"
        if candidate.exists() and candidate.is_file():
            return candidate

        candidate_upper = IMAGE_DIR / f"{stem}{ext.upper()}"
        if candidate_upper.exists() and candidate_upper.is_file():
            return candidate_upper

    matches = list(IMAGE_DIR.glob(f"{stem}.*"))
    for path in matches:
        if path.is_file():
            return path

    return None


# ============================================================
# FG-CLIP2 文本编码
# ============================================================

def encode_text_batch(text_list):
    """
    输入 List[str]
    输出 [B, 1024]
    """
    inputs = tokenizer(
        text_list,
        padding="max_length",
        max_length=TEXT_MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    )

    inputs = move_to_device(inputs)

    with torch.inference_mode():
        features = model.get_text_features(
            **inputs,
            walk_type=TEXT_WALK_TYPE,
        )

    if L2_NORMALIZE:
        features = normalize_feature(features)

    if features.ndim != 2 or features.shape[-1] != FEATURE_DIM:
        raise ValueError(
            f"文本特征维度错误：得到 {tuple(features.shape)}，"
            f"期望 [B, {FEATURE_DIM}]"
        )

    return features


# ============================================================
# FG-CLIP2 图片编码
# ============================================================

def encode_image(image_path):
    """
    输入单张图片路径
    输出 [1, 1024]
    """
    with Image.open(image_path) as image:

        # GIF 默认取第一帧
        if getattr(image, "is_animated", False):
            try:
                image.seek(0)
            except Exception:
                pass

        image = image.convert("RGB")

        max_num_patches = determine_max_num_patches(image)

        inputs = image_processor(
            images=image,
            max_num_patches=max_num_patches,
            return_tensors="pt",
        )

    inputs = move_to_device(inputs)

    with torch.inference_mode():
        feature = model.get_image_features(**inputs)

    if L2_NORMALIZE:
        feature = normalize_feature(feature)

    if feature.ndim != 2 or feature.shape[-1] != FEATURE_DIM:
        raise ValueError(
            f"图片特征维度错误：得到 {tuple(feature.shape)}，"
            f"期望 [1, {FEATURE_DIM}]"
        )

    return feature


# ============================================================
# 批量执行文本编码
# ============================================================

def run_text_records(records, progress_name):
    """
    records:
        [
            (text, save_path),
            ...
        ]
    """
    if not records:
        print(f"{progress_name}：没有需要编码的文本。")
        return

    print(f"{progress_name}：需要编码 {len(records)} 条。")

    for start in tqdm(
        range(0, len(records), TEXT_BATCH_SIZE),
        desc=progress_name,
    ):
        batch_records = records[start:start + TEXT_BATCH_SIZE]
        batch_texts = [x[0] for x in batch_records]
        batch_paths = [x[1] for x in batch_records]

        try:
            features = encode_text_batch(batch_texts)

            for feature, save_path in zip(features, batch_paths):
                save_feature(feature, save_path)

        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise

            print(
                "\n检测到 GPU 显存不足，当前批次自动退化为逐条编码。"
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for text, save_path in batch_records:
                try:
                    feature = encode_text_batch([text])[0]
                    save_feature(feature, save_path)
                except Exception as sub_e:
                    print(
                        f"\n[文本编码失败] 文件={save_path.name}\n"
                        f"原因：{sub_e}"
                    )


# ============================================================
# 读取两个 Excel
# ============================================================

print("\n" + "=" * 90)
print("读取 Weibo 数据")
print("=" * 90)

text_df = pd.read_excel(
    TEXT_EXCEL_PATH,
    dtype=str,
    keep_default_na=False,
)

desc_df = pd.read_excel(
    DESC_EXCEL_PATH,
    dtype=str,
    keep_default_na=False,
)


# ============================================================
# 检查列结构
# ============================================================

required_text_cols = {"id", "text"}
required_desc_cols = {"id", "image_name", "description"}

if not required_text_cols.issubset(text_df.columns):
    raise ValueError(
        f"{TEXT_EXCEL_PATH.name} 列结构不正确。\n"
        f"实际列：{list(text_df.columns)}\n"
        f"至少需要：id, text"
    )

if not required_desc_cols.issubset(desc_df.columns):
    raise ValueError(
        f"{DESC_EXCEL_PATH.name} 列结构不正确。\n"
        f"实际列：{list(desc_df.columns)}\n"
        f"至少需要：id, image_name, description"
    )


# ============================================================
# 清理 ID，并检查文本表是否一 ID 一条
# ============================================================

text_df["id"] = text_df["id"].astype(str).str.strip()
desc_df["id"] = desc_df["id"].astype(str).str.strip()

if (text_df["id"] == "").any():
    raise ValueError("weibo_text.xlsx 中存在空 id。")

if (desc_df["id"] == "").any():
    raise ValueError("weibo_image_description.xlsx 中存在空 id。")

duplicate_text_ids = text_df.loc[
    text_df["id"].duplicated(keep=False),
    "id"
].unique()

if len(duplicate_text_ids) > 0:
    raise ValueError(
        "weibo_text.xlsx 中存在重复 id。为了避免 id_T.npy 被覆盖，"
        f"请先检查这些 id，例如：{duplicate_text_ids[:20].tolist()}"
    )


# ============================================================
# 为同一个 ID 的多张图片按 Excel 原始顺序编号
#
# 例如某 ID 有 3 行：
#   第1行 -> index 0 -> id_I_0.npy / id_C_0.npy
#   第2行 -> index 1 -> id_I_1.npy / id_C_1.npy
#   第3行 -> index 2 -> id_I_2.npy / id_C_2.npy
# ============================================================

desc_df["media_index"] = (
    desc_df.groupby("id", sort=False).cumcount()
)


# ============================================================
# 数据对应关系检查
# ============================================================

text_ids = set(text_df["id"])
desc_ids = set(desc_df["id"])

text_only_ids = sorted(text_ids - desc_ids)
desc_only_ids = sorted(desc_ids - text_ids)

print(f"文本样本数：{len(text_df)}")
print(f"图片/描述记录数：{len(desc_df)}")
print(f"文本唯一 ID 数：{text_df['id'].nunique()}")
print(f"图片/描述唯一 ID 数：{desc_df['id'].nunique()}")
print(f"单个 ID 最大图片数：{desc_df.groupby('id').size().max()}")

if text_only_ids:
    print(f"警告：有 {len(text_only_ids)} 个 ID 只有文本，没有图片/描述。")

if desc_only_ids:
    print(f"警告：有 {len(desc_only_ids)} 个 ID 只有图片/描述，没有文本。")


# ============================================================
# 保存一个索引清单，明确每个多图编号对应哪张原图
# ============================================================

manifest_df = desc_df[
    ["id", "media_index", "image_name", "description"]
].copy()

manifest_df["image_feature"] = (
    manifest_df["id"]
    + "_I_"
    + manifest_df["media_index"].astype(str)
    + ".npy"
)

manifest_df["description_feature"] = (
    manifest_df["id"]
    + "_C_"
    + manifest_df["media_index"].astype(str)
    + ".npy"
)

manifest_path = OUTPUT_DIR / "weibo_feature_manifest.xlsx"
manifest_df.to_excel(manifest_path, index=False)

print(f"多图编号映射清单：{manifest_path}")


# ============================================================
# 第一阶段：Text -> id_T.npy
# ============================================================

print("\n" + "=" * 90)
print("第一阶段：提取 Text 特征")
print("=" * 90)

text_records = []

for row in text_df.itertuples(index=False):
    sample_id = str(row.id).strip()
    text = clean_text(row.text)

    save_path = OUTPUT_DIR / f"{sample_id}_T.npy"

    if not OVERWRITE and save_path.exists():
        continue

    if text:
        text_records.append((text, save_path))
    else:
        print(f"[Text 缺失] ID={sample_id}，保存零向量。")
        save_zero_feature(save_path)

run_text_records(
    text_records,
    progress_name="Text",
)


# ============================================================
# 第二阶段：Description -> id_C_0.npy / id_C_1.npy / ...
# ============================================================

print("\n" + "=" * 90)
print("第二阶段：提取 Description 特征")
print("=" * 90)

description_records = []

for row in desc_df.itertuples(index=False):
    sample_id = str(row.id).strip()
    media_index = int(row.media_index)
    description = clean_text(row.description)

    save_path = OUTPUT_DIR / f"{sample_id}_C_{media_index}.npy"

    if not OVERWRITE and save_path.exists():
        continue

    if description:
        description_records.append((description, save_path))
    else:
        print(
            f"[Description 缺失] ID={sample_id}, index={media_index}，"
            "保存零向量。"
        )
        save_zero_feature(save_path)

run_text_records(
    description_records,
    progress_name="Description",
)


# ============================================================
# 第三阶段：Image -> id_I_0.npy / id_I_1.npy / ...
#
# 图片和 description 使用同一个 media_index，
# 因此：
#   id_I_0.npy <-> id_C_0.npy
#   id_I_1.npy <-> id_C_1.npy
# ============================================================

print("\n" + "=" * 90)
print("第三阶段：提取 Image 特征")
print("=" * 90)

missing_images = []
failed_images = []

for row in tqdm(
    desc_df.itertuples(index=False),
    total=len(desc_df),
    desc="Image",
):
    sample_id = str(row.id).strip()
    media_index = int(row.media_index)
    image_name = str(row.image_name).strip()

    save_path = OUTPUT_DIR / f"{sample_id}_I_{media_index}.npy"

    if not OVERWRITE and save_path.exists():
        continue

    image_path = find_image(image_name)

    if image_path is None:
        print(
            f"\n[图片不存在] ID={sample_id}, index={media_index}, "
            f"image={image_name}"
        )

        missing_images.append({
            "id": sample_id,
            "media_index": media_index,
            "image_name": image_name,
            "expected_feature": save_path.name,
        })

        # 保证 I_k 与 C_k 编号一一对应
        save_zero_feature(save_path)
        continue

    try:
        feature = encode_image(image_path)[0]
        save_feature(feature, save_path)

    except Exception as e:
        print(
            f"\n[图片编码失败]\n"
            f"ID：{sample_id}\n"
            f"index：{media_index}\n"
            f"图片：{image_path}\n"
            f"错误：{e}"
        )

        failed_images.append({
            "id": sample_id,
            "media_index": media_index,
            "image_name": image_name,
            "image_path": str(image_path),
            "expected_feature": save_path.name,
            "error": str(e),
        })

        save_zero_feature(save_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# 保存错误日志
# ============================================================

if missing_images:
    pd.DataFrame(missing_images).to_excel(
        OUTPUT_DIR / "missing_images.xlsx",
        index=False,
    )

if failed_images:
    pd.DataFrame(failed_images).to_excel(
        OUTPUT_DIR / "failed_images.xlsx",
        index=False,
    )


# ============================================================
# 最终完整性检查
# ============================================================

print("\n" + "=" * 90)
print("开始检查全部特征文件")
print("=" * 90)

bad_feature_files = []
missing_feature_files = []

expected_t = []
expected_i = []
expected_c = []

# T
for sample_id in text_df["id"]:
    expected_t.append(OUTPUT_DIR / f"{sample_id}_T.npy")

# I / C
for row in desc_df.itertuples(index=False):
    sample_id = str(row.id).strip()
    media_index = int(row.media_index)

    expected_i.append(
        OUTPUT_DIR / f"{sample_id}_I_{media_index}.npy"
    )

    expected_c.append(
        OUTPUT_DIR / f"{sample_id}_C_{media_index}.npy"
    )


def check_feature_files(paths, feature_type):
    valid_count = 0

    for path in paths:
        if not path.exists():
            missing_feature_files.append(
                (feature_type, path.name)
            )
            continue

        try:
            feature = np.load(path)

            if feature.shape != (FEATURE_DIM,):
                bad_feature_files.append(
                    (
                        feature_type,
                        path.name,
                        f"shape={feature.shape}",
                    )
                )
            elif feature.dtype != np.float32:
                bad_feature_files.append(
                    (
                        feature_type,
                        path.name,
                        f"dtype={feature.dtype}",
                    )
                )
            else:
                valid_count += 1

        except Exception as e:
            bad_feature_files.append(
                (
                    feature_type,
                    path.name,
                    str(e),
                )
            )

    return valid_count


text_count = check_feature_files(expected_t, "T")
image_count = check_feature_files(expected_i, "I")
description_count = check_feature_files(expected_c, "C")


# ============================================================
# 输出结果
# ============================================================

print("\n" + "=" * 90)
print("FG-CLIP2-Large Weibo 特征提取完成")
print("=" * 90)

print(
    f"Text：{text_count}/{len(expected_t)} "
    f"-> id_T.npy"
)

print(
    f"Image：{image_count}/{len(expected_i)} "
    f"-> id_I_0.npy, id_I_1.npy, ..."
)

print(
    f"Description：{description_count}/{len(expected_c)} "
    f"-> id_C_0.npy, id_C_1.npy, ..."
)

print(f"特征维度：{FEATURE_DIM}")
print("特征格式：float32")
print(f"L2 归一化：{L2_NORMALIZE}")
print(f"输出目录：{OUTPUT_DIR}")

if missing_images:
    print(f"未找到原图：{len(missing_images)}")
    print(f"日志：{OUTPUT_DIR / 'missing_images.xlsx'}")

if failed_images:
    print(f"图片编码失败：{len(failed_images)}")
    print(f"日志：{OUTPUT_DIR / 'failed_images.xlsx'}")

if missing_feature_files:
    print(
        f"最终仍缺少特征文件：{len(missing_feature_files)}"
    )
    for item in missing_feature_files[:20]:
        print("  ", item)

if bad_feature_files:
    print(
        f"发现异常特征文件：{len(bad_feature_files)}"
    )
    for item in bad_feature_files[:20]:
        print("  ", item)
else:
    print("所有已生成特征均为 float32、1024 维。")

print("\n命名规则：")
print("  文本：id_T.npy")
print("  图片：id_I_0.npy, id_I_1.npy, id_I_2.npy, ...")
print("  描述：id_C_0.npy, id_C_1.npy, id_C_2.npy, ...")
print("\n同一个 ID 下，I_k 与 C_k 严格对应同一行 image_name / description。")
print("=" * 90)
