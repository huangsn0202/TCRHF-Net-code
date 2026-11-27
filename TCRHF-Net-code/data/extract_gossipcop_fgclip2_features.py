"""
使用 FG-CLIP2-Large 为 GossipCop 提取 1024 维全局特征。

当前代码保存位置：
    D:\TIC_code\data\extract_gossipcop_fgclip2_features.py

数据文件：
    D:\TIC_code\data\goosipcop\gossipcop_all_data.xlsx

图片目录：
    D:\TIC_code\data\goosipcop\goosipcop_image\

模型目录：
    D:\TIC_code\FG-CLIP2-Large\

输出目录：
    D:\TIC_code\data\feature\goosipcop\

输出命名：
    文本：      id_T.npy
    图片：      id_I.npy
    图片描述：  id_C.npy

所有特征均保存为：
    shape = (1024,)
    dtype = float32

说明：
1. 提取方式与 extract_weibo_fgclip2_features.py 保持一致；
2. 文本和图片描述使用 FG-CLIP2-Large 的 get_text_features；
3. 图片使用 FG-CLIP2-Large 的 get_image_features；
4. 默认对最终 1024 维特征做 L2 归一化；
5. 支持断点续跑，已经存在且维度正确的特征默认跳过；
6. GossipCop 当前按“一条 ID 对应一张图片和一条图片描述”处理。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
# D:\TIC_code\data\extract_gossipcop_fgclip2_features.py
#
# 数据文件：
# D:\TIC_code\data\goosipcop\gossipcop_all_data.xlsx
#
# 图片：
# D:\TIC_code\data\goosipcop\goosipcop_image\
#
# 模型：
# D:\TIC_code\FG-CLIP2-Large\
#
# 输出：
# D:\TIC_code\data\feature\goosipcop\
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

EXCEL_PATH = BASE_DIR / "goosipcop" / "gossipcop_all_data.xlsx"
IMAGE_DIR = BASE_DIR / "goosipcop" / "goosipcop_image"

MODEL_DIR = BASE_DIR.parent / "FG-CLIP2-Large"

# 关键修改：不再保存到 goosipcop-xi
OUTPUT_DIR = BASE_DIR / "feature" / "goosipcop"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 参数配置
# ============================================================

# FG-CLIP2-Large long-text 模式
TEXT_MAX_LENGTH = 196
TEXT_WALK_TYPE = "long"

# 与 Weibo 特征提取代码保持一致
TEXT_BATCH_SIZE = 32

# True  = 保存 L2 归一化后的 1024 维特征
# False = 保存模型原始 1024 维特征
L2_NORMALIZE = True

# False：断点续跑，已存在且正确的 .npy 不重复计算
# True ：覆盖并重新提取全部特征
OVERWRITE = False

FEATURE_DIM = 1024

# 与 Weibo 代码一致：文本统一转小写。
LOWERCASE_TEXT = True


# ============================================================
# 图片格式
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
# 基础路径检查
# ============================================================

for path, name in [
    (EXCEL_PATH, "gossipcop_all_data.xlsx"),
    (IMAGE_DIR, "goosipcop_image 文件夹"),
    (MODEL_DIR, "FG-CLIP2-Large 模型文件夹"),
]:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {name}：\n{path}"
        )


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

def normalize_feature(feature: torch.Tensor) -> torch.Tensor:
    """
    对最后一维做 L2 归一化。
    """
    norm = torch.norm(
        feature,
        p=2,
        dim=-1,
        keepdim=True,
    )
    return feature / torch.clamp(norm, min=1e-12)


def determine_max_num_patches(image: Image.Image) -> int:
    """
    与 extract_weibo_fgclip2_features.py 保持一致的动态 patch 策略。
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


def move_to_device(inputs: Dict[str, object]) -> Dict[str, object]:
    """
    把 tokenizer / image processor 中的 Tensor 移到设备。
    """
    result = {}

    for key, value in inputs.items():
        if torch.is_tensor(value):
            result[key] = value.to(DEVICE)
        else:
            result[key] = value

    return result


def clean_text(text: object) -> str:
    """
    文本清理，与 Weibo 特征提取代码保持一致。
    """
    if text is None:
        return ""

    text = str(text).strip()

    if text.lower() in {"", "nan", "none", "null", "na"}:
        return ""

    # 合并连续空白
    text = " ".join(text.split())

    if LOWERCASE_TEXT:
        text = text.lower()

    return text


def save_feature(
    feature: torch.Tensor,
    save_path: Path,
) -> None:
    """
    保存为 shape=(1024,) 的 float32 npy。
    """
    feature = (
        feature
        .detach()
        .float()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    if feature.shape != (FEATURE_DIM,):
        raise ValueError(
            f"特征维度错误：{save_path.name}，"
            f"得到 {feature.shape}，"
            f"期望 ({FEATURE_DIM},)"
        )

    if not np.isfinite(feature).all():
        raise ValueError(
            f"特征中存在 NaN 或 Inf：{save_path.name}"
        )

    np.save(
        save_path,
        feature.astype(np.float32),
    )


def save_zero_feature(save_path: Path) -> None:
    """
    缺失模态保存 1024 维零向量。
    """
    np.save(
        save_path,
        np.zeros(
            FEATURE_DIM,
            dtype=np.float32,
        ),
    )


def feature_file_is_valid(path: Path) -> bool:
    """
    用于断点续跑：
    只有文件存在、shape=(1024,)、dtype=float32 且无 NaN/Inf，
    才视为已经完成。
    """
    if not path.exists():
        return False

    try:
        feature = np.load(
            path,
            mmap_mode="r",
            allow_pickle=False,
        )

        if feature.shape != (FEATURE_DIM,):
            return False

        if feature.dtype != np.float32:
            return False

        # mmap 下仍可检查有限性
        if not np.isfinite(feature).all():
            return False

        return True

    except Exception:
        return False


# ============================================================
# 图片路径索引
# ============================================================

def build_image_index() -> Dict[str, Path]:
    """
    对图片目录建立索引，避免每个样本反复 glob 整个目录。

    同时支持：
    1. 完整文件名；
    2. 去掉后缀后的 stem；
    3. 大小写差异。
    """
    image_index: Dict[str, Path] = {}

    for path in IMAGE_DIR.iterdir():

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image_index.setdefault(
            path.name.lower(),
            path,
        )

        image_index.setdefault(
            path.stem.lower(),
            path,
        )

    return image_index


def find_image(
    image_name: object,
    image_index: Dict[str, Path],
) -> Optional[Path]:
    """
    根据 Excel 中的图片名称查找原图。
    """
    image_name = str(image_name).strip()

    if not image_name:
        return None

    # 先按完整文件名查找
    path = image_index.get(
        image_name.lower()
    )

    if path is not None:
        return path

    # 再按 stem 查找
    stem = Path(image_name).stem.lower()

    return image_index.get(stem)


# ============================================================
# FG-CLIP2 文本编码
# ============================================================

def encode_text_batch(
    text_list: Sequence[str],
) -> torch.Tensor:
    """
    输入：
        List[str]

    输出：
        [B, 1024]

    与 Weibo 代码一致，直接使用 FG-CLIP2-Large
    的全局文本特征 get_text_features。
    """
    inputs = tokenizer(
        list(text_list),
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

    if (
        features.ndim != 2
        or features.shape[-1] != FEATURE_DIM
    ):
        raise ValueError(
            f"文本特征维度错误："
            f"得到 {tuple(features.shape)}，"
            f"期望 [B, {FEATURE_DIM}]"
        )

    return features


# ============================================================
# FG-CLIP2 图片编码
# ============================================================

def encode_image(
    image_path: Path,
) -> torch.Tensor:
    """
    输入：
        单张图片路径

    输出：
        [1, 1024]

    与 Weibo 代码一致，直接使用 FG-CLIP2-Large
    的全局图片特征 get_image_features。
    """
    with Image.open(image_path) as image:

        # GIF 默认使用第一帧
        if getattr(image, "is_animated", False):
            try:
                image.seek(0)
            except Exception:
                pass

        image = image.convert("RGB")

        max_num_patches = determine_max_num_patches(
            image
        )

        inputs = image_processor(
            images=image,
            max_num_patches=max_num_patches,
            return_tensors="pt",
        )

    inputs = move_to_device(inputs)

    with torch.inference_mode():
        feature = model.get_image_features(
            **inputs
        )

    if L2_NORMALIZE:
        feature = normalize_feature(feature)

    if (
        feature.ndim != 2
        or feature.shape[-1] != FEATURE_DIM
    ):
        raise ValueError(
            f"图片特征维度错误："
            f"得到 {tuple(feature.shape)}，"
            f"期望 [1, {FEATURE_DIM}]"
        )

    return feature


# ============================================================
# 批量文本编码
# ============================================================

TextRecord = Tuple[str, Path]


def run_text_records(
    records: Sequence[TextRecord],
    progress_name: str,
) -> List[Dict[str, str]]:
    """
    records:
        [
            (text, save_path),
            ...
        ]

    返回失败日志。
    """
    failures: List[Dict[str, str]] = []

    if not records:
        print(
            f"{progress_name}：没有需要编码的文本。"
        )
        return failures

    print(
        f"{progress_name}：需要编码 "
        f"{len(records)} 条。"
    )

    for start in tqdm(
        range(
            0,
            len(records),
            TEXT_BATCH_SIZE,
        ),
        desc=progress_name,
    ):
        batch_records = records[
            start:start + TEXT_BATCH_SIZE
        ]

        batch_texts = [
            item[0]
            for item in batch_records
        ]

        batch_paths = [
            item[1]
            for item in batch_records
        ]

        try:
            features = encode_text_batch(
                batch_texts
            )

            for feature, save_path in zip(
                features,
                batch_paths,
            ):
                save_feature(
                    feature,
                    save_path,
                )

            continue

        except RuntimeError as error:

            if "out of memory" not in str(error).lower():
                print(
                    f"\n[{progress_name} 批量编码失败] "
                    f"{error}"
                )
            else:
                print(
                    "\n检测到 GPU 显存不足，"
                    "当前批次自动退化为逐条编码。"
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as error:
            print(
                f"\n[{progress_name} 批量编码失败] "
                f"{error}\n"
                "当前批次自动退化为逐条编码。"
            )

        # 批量失败后逐条重试
        for text, save_path in batch_records:

            try:
                feature = encode_text_batch(
                    [text]
                )[0]

                save_feature(
                    feature,
                    save_path,
                )

            except Exception as sub_error:

                print(
                    f"\n[{progress_name} 编码失败]\n"
                    f"文件：{save_path.name}\n"
                    f"错误：{sub_error}"
                )

                failures.append(
                    {
                        "feature_file": save_path.name,
                        "error": str(sub_error),
                    }
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return failures


# ============================================================
# 读取 GossipCop Excel
# ============================================================

print("\n" + "=" * 90)
print("读取 GossipCop 数据")
print(f"Excel：{EXCEL_PATH}")
print("=" * 90)

dataframe = pd.read_excel(
    EXCEL_PATH,
    dtype=str,
    keep_default_na=False,
)


# ============================================================
# 数据列结构检查
#
# 原 GossipCop 文件按以下顺序读取：
# 第 1 列：id
# 第 2 列：image name
# 第 3 列：label
# 第 4 列：text
# 第 5 列：description
# ============================================================

if dataframe.shape[1] < 5:
    raise ValueError(
        "gossipcop_all_data.xlsx 至少需要 5 列，"
        "且顺序必须为：\n"
        "id, image name, label, text, description"
    )

sample_ids = (
    dataframe.iloc[:, 0]
    .astype(str)
    .str.strip()
)

if (sample_ids == "").any():
    rows = (
        sample_ids[
            sample_ids == ""
        ].index + 2
    ).tolist()[:20]

    raise ValueError(
        f"发现空 ID，Excel 行号例如：{rows}"
    )

duplicate_ids = (
    sample_ids[
        sample_ids.duplicated(
            keep=False
        )
    ]
    .unique()
    .tolist()
)

if duplicate_ids:
    raise ValueError(
        "GossipCop 中存在重复 ID，"
        "会导致特征文件互相覆盖，例如："
        f"{duplicate_ids[:20]}"
    )

if sample_ids.str.contains(
    r"[\\/]",
    regex=True,
).any():
    invalid_ids = (
        sample_ids[
            sample_ids.str.contains(
                r"[\\/]",
                regex=True,
            )
        ]
        .tolist()[:20]
    )

    raise ValueError(
        "ID 中不能包含路径分隔符，"
        f"例如：{invalid_ids}"
    )

print(f"样本总数：{len(dataframe)}")


# ============================================================
# 第一阶段：Text -> id_T.npy
# ============================================================

print("\n" + "=" * 90)
print("第一阶段：提取 Text 特征")
print("=" * 90)

text_records: List[TextRecord] = []

for row_idx in range(len(dataframe)):

    sample_id = str(
        dataframe.iloc[row_idx, 0]
    ).strip()

    text = clean_text(
        dataframe.iloc[row_idx, 3]
    )

    save_path = (
        OUTPUT_DIR
        / f"{sample_id}_T.npy"
    )

    if (
        not OVERWRITE
        and feature_file_is_valid(
            save_path
        )
    ):
        continue

    if text:
        text_records.append(
            (
                text,
                save_path,
            )
        )
    else:
        print(
            f"[Text 缺失] "
            f"ID={sample_id}，"
            "保存 1024 维零向量。"
        )

        save_zero_feature(
            save_path
        )

text_failures = run_text_records(
    text_records,
    progress_name="Text",
)


# ============================================================
# 第二阶段：Description -> id_C.npy
# ============================================================

print("\n" + "=" * 90)
print("第二阶段：提取 Description 特征")
print("=" * 90)

description_records: List[TextRecord] = []

for row_idx in range(len(dataframe)):

    sample_id = str(
        dataframe.iloc[row_idx, 0]
    ).strip()

    description = clean_text(
        dataframe.iloc[row_idx, 4]
    )

    save_path = (
        OUTPUT_DIR
        / f"{sample_id}_C.npy"
    )

    if (
        not OVERWRITE
        and feature_file_is_valid(
            save_path
        )
    ):
        continue

    if description:
        description_records.append(
            (
                description,
                save_path,
            )
        )
    else:
        print(
            f"[Description 缺失] "
            f"ID={sample_id}，"
            "保存 1024 维零向量。"
        )

        save_zero_feature(
            save_path
        )

description_failures = run_text_records(
    description_records,
    progress_name="Description",
)


# ============================================================
# 第三阶段：Image -> id_I.npy
# ============================================================

print("\n" + "=" * 90)
print("第三阶段：提取 Image 特征")
print("=" * 90)

image_index = build_image_index()

print(
    f"图片目录索引完成，"
    f"索引键数量：{len(image_index)}"
)

missing_images: List[Dict[str, str]] = []
failed_images: List[Dict[str, str]] = []

for row_idx in tqdm(
    range(len(dataframe)),
    desc="Image",
):

    sample_id = str(
        dataframe.iloc[row_idx, 0]
    ).strip()

    image_name = str(
        dataframe.iloc[row_idx, 1]
    ).strip()

    save_path = (
        OUTPUT_DIR
        / f"{sample_id}_I.npy"
    )

    if (
        not OVERWRITE
        and feature_file_is_valid(
            save_path
        )
    ):
        continue

    image_path = find_image(
        image_name,
        image_index,
    )

    if image_path is None:

        print(
            f"\n[图片不存在] "
            f"ID={sample_id}, "
            f"image={image_name}"
        )

        missing_images.append(
            {
                "id": sample_id,
                "image_name": image_name,
                "expected_feature": save_path.name,
            }
        )

        # 保证每个 ID 都有 I 文件
        save_zero_feature(
            save_path
        )

        continue

    try:
        feature = encode_image(
            image_path
        )[0]

        save_feature(
            feature,
            save_path,
        )

    except Exception as error:

        print(
            f"\n[图片编码失败]\n"
            f"ID：{sample_id}\n"
            f"图片：{image_path}\n"
            f"错误：{error}"
        )

        failed_images.append(
            {
                "id": sample_id,
                "image_name": image_name,
                "image_path": str(
                    image_path
                ),
                "expected_feature": save_path.name,
                "error": str(error),
            }
        )

        # 与 Weibo 代码保持一致：
        # 失败时保存零向量，保证特征文件对应关系不乱。
        save_zero_feature(
            save_path
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# 保存错误日志
# ============================================================

if missing_images:
    pd.DataFrame(
        missing_images
    ).to_excel(
        OUTPUT_DIR
        / "missing_images.xlsx",
        index=False,
    )

if failed_images:
    pd.DataFrame(
        failed_images
    ).to_excel(
        OUTPUT_DIR
        / "failed_images.xlsx",
        index=False,
    )

all_text_failures = (
    text_failures
    + description_failures
)

if all_text_failures:
    pd.DataFrame(
        all_text_failures
    ).to_excel(
        OUTPUT_DIR
        / "failed_text.xlsx",
        index=False,
    )


# ============================================================
# 最终完整性检查
# ============================================================

print("\n" + "=" * 90)
print("开始检查全部特征文件")
print("=" * 90)

bad_feature_files: List[
    Tuple[str, str, str]
] = []

missing_feature_files: List[
    Tuple[str, str]
] = []


def check_feature_files(
    paths: Sequence[Path],
    feature_type: str,
) -> int:

    valid_count = 0

    for path in paths:

        if not path.exists():
            missing_feature_files.append(
                (
                    feature_type,
                    path.name,
                )
            )
            continue

        try:
            feature = np.load(
                path,
                allow_pickle=False,
            )

            if feature.shape != (
                FEATURE_DIM,
            ):
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

            elif not np.isfinite(
                feature
            ).all():
                bad_feature_files.append(
                    (
                        feature_type,
                        path.name,
                        "contains NaN or Inf",
                    )
                )

            else:
                valid_count += 1

        except Exception as error:
            bad_feature_files.append(
                (
                    feature_type,
                    path.name,
                    str(error),
                )
            )

    return valid_count


expected_t = [
    OUTPUT_DIR
    / f"{sample_id}_T.npy"
    for sample_id in sample_ids
]

expected_i = [
    OUTPUT_DIR
    / f"{sample_id}_I.npy"
    for sample_id in sample_ids
]

expected_c = [
    OUTPUT_DIR
    / f"{sample_id}_C.npy"
    for sample_id in sample_ids
]

text_count = check_feature_files(
    expected_t,
    "T",
)

image_count = check_feature_files(
    expected_i,
    "I",
)

description_count = check_feature_files(
    expected_c,
    "C",
)


# ============================================================
# 输出结果
# ============================================================

print("\n" + "=" * 90)
print("FG-CLIP2-Large GossipCop 特征提取完成")
print("=" * 90)

print(
    f"Text：{text_count}/"
    f"{len(expected_t)} "
    "-> id_T.npy"
)

print(
    f"Image：{image_count}/"
    f"{len(expected_i)} "
    "-> id_I.npy"
)

print(
    f"Description：{description_count}/"
    f"{len(expected_c)} "
    "-> id_C.npy"
)

print(f"特征维度：{FEATURE_DIM}")
print("特征格式：float32")
print(f"L2 归一化：{L2_NORMALIZE}")
print(f"输出目录：{OUTPUT_DIR}")

if missing_images:
    print(
        f"未找到原图："
        f"{len(missing_images)}"
    )
    print(
        "日志："
        f"{OUTPUT_DIR / 'missing_images.xlsx'}"
    )

if failed_images:
    print(
        f"图片编码失败："
        f"{len(failed_images)}"
    )
    print(
        "日志："
        f"{OUTPUT_DIR / 'failed_images.xlsx'}"
    )

if all_text_failures:
    print(
        f"文本/描述编码失败："
        f"{len(all_text_failures)}"
    )
    print(
        "日志："
        f"{OUTPUT_DIR / 'failed_text.xlsx'}"
    )

if missing_feature_files:
    print(
        f"最终仍缺少特征文件："
        f"{len(missing_feature_files)}"
    )

    for item in missing_feature_files[:20]:
        print("  ", item)

if bad_feature_files:
    print(
        f"发现异常特征文件："
        f"{len(bad_feature_files)}"
    )

    for item in bad_feature_files[:20]:
        print("  ", item)

else:
    print(
        "所有已生成特征均为 "
        "float32、1024 维。"
    )

print("\n命名规则：")
print("  文本：id_T.npy")
print("  图片：id_I.npy")
print("  描述：id_C.npy")
print("=" * 90)
