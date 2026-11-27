from pathlib import Path
import sys
import random
import re

import pandas as pd
from transformers import AutoTokenizer


# ============================================================
# Windows 控制台尽量使用 UTF-8
# 防止“数据本身正常，但 CMD/PyCharm 控制台显示乱码”
# ============================================================

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ============================================================
# 路径
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

EXCEL_PATH = (
    BASE_DIR
    / "goosipcop"
    / "gossipcop_all_data.xlsx"
)

MODEL_DIR = (
    BASE_DIR.parent
    / "FG-CLIP2-Large"
)

# 检查结果
OUTPUT_XLSX = (
    BASE_DIR
    / "FGCLIP2_文本输入检查.xlsx"
)

OUTPUT_TXT = (
    BASE_DIR
    / "FGCLIP2_文本输入检查.txt"
)


# ============================================================
# 参数
# ============================================================

# 与正式特征提取代码保持完全一致
TEXT_MAX_LENGTH = 196

# 与正式代码保持一致
LOWERCASE_TEXT = True

# 随机检查多少个 ID
NUM_SAMPLES = 20

# 固定随机种子，保证重复运行抽到同一批
RANDOM_SEED = 42


# ============================================================
# 如果你想指定某几个 ID 检查
#
# 例如：
# CHECK_IDS = ["123", "456", "789"]
#
# 留空：
# CHECK_IDS = []
#
# 就会自动随机抽取 NUM_SAMPLES 个样本
# ============================================================

CHECK_IDS = []


# ============================================================
# 文本清洗
# 与你的正式特征提取代码保持一致
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text).strip()

    if text.lower() in [
        "nan",
        "none",
        "null",
        "na"
    ]:
        return ""

    # 合并连续空白
    text = " ".join(text.split())

    # 正式代码中当前设置为 True
    if LOWERCASE_TEXT:
        text = text.lower()

    return text


# ============================================================
# 用于比较文字
# 这里只用于检查，不参与模型
# ============================================================

def normalize_for_compare(text):

    text = str(text)

    # 合并空格
    text = " ".join(text.split())

    # 转小写
    text = text.lower()

    return text


# ============================================================
# 检查明显乱码字符
# ============================================================

def detect_suspicious_characters(text):

    text = str(text)

    problems = []

    # Unicode replacement character
    if "\ufffd" in text:
        problems.append("发现 � 字符")

    # NULL 字符
    if "\x00" in text:
        problems.append("发现 NULL 字符")

    # 一些常见 UTF-8 / Latin-1 错误解码表现
    patterns = [
        "Ã",
        "Â",
        "â€",
        "â€™",
        "â€œ",
        "â€˜",
        "ðŸ"
    ]

    for p in patterns:
        if p in text:
            problems.append(
                f"疑似乱码模式: {p}"
            )

    if problems:
        return "；".join(problems)

    return "未发现明显乱码"


# ============================================================
# 加载 tokenizer
# ============================================================

print("=" * 90)
print("加载 FG-CLIP2-Large tokenizer")
print("=" * 90)

if not MODEL_DIR.exists():

    raise FileNotFoundError(
        f"模型目录不存在：\n{MODEL_DIR}"
    )


tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR),
    trust_remote_code=True,
    local_files_only=True
)

print(f"Tokenizer 加载成功：{MODEL_DIR}")


# ============================================================
# 读取 Excel
# ============================================================

print("\n" + "=" * 90)
print("读取 GossipCop Excel")
print("=" * 90)

if not EXCEL_PATH.exists():

    raise FileNotFoundError(
        f"Excel 不存在：\n{EXCEL_PATH}"
    )


df = pd.read_excel(
    EXCEL_PATH,
    dtype=str,
    keep_default_na=False
)


if df.shape[1] < 5:

    raise ValueError(
        "Excel 至少需要 5 列：\n"
        "第0列 id\n"
        "第1列 image\n"
        "第2列 label\n"
        "第3列 text\n"
        "第4列 description"
    )


print(f"样本数量：{len(df)}")

print("\n列名：")

for i, col in enumerate(df.columns):
    print(f"第{i}列：{col}")


# ============================================================
# ID 全部转换成字符串
# ============================================================

df.iloc[:, 0] = (
    df.iloc[:, 0]
    .astype(str)
    .str.strip()
)


# ============================================================
# 选择需要检查的样本
# ============================================================

if CHECK_IDS:

    check_id_set = set(
        str(x).strip()
        for x in CHECK_IDS
    )

    sample_df = df[
        df.iloc[:, 0].isin(check_id_set)
    ].copy()

    missing_ids = (
        check_id_set
        -
        set(sample_df.iloc[:, 0].tolist())
    )

    if missing_ids:

        print(
            "\n以下指定 ID 在 Excel 中没有找到："
        )

        for x in missing_ids:
            print(x)

else:

    sample_num = min(
        NUM_SAMPLES,
        len(df)
    )

    sample_df = df.sample(
        n=sample_num,
        random_state=RANDOM_SEED
    ).copy()


print(
    f"\n本次检查样本数："
    f"{len(sample_df)}"
)


# ============================================================
# 检查单条文本
# ============================================================

def inspect_text(
    sample_id,
    text_type,
    raw_text
):

    cleaned_text = clean_text(
        raw_text
    )

    # --------------------------------------------------------
    # 空文本
    # --------------------------------------------------------

    if not cleaned_text:

        return {
            "id": sample_id,
            "类型": text_type,
            "Excel原始文字": raw_text,
            "clean_text后文字": cleaned_text,
            "模型实际decode文字": "",
            "截断前token数量": 0,
            "实际输入token数量": 0,
            "是否超过196": False,
            "是否被截断": False,
            "文字一致性": "空文本",
            "乱码检查_原文": detect_suspicious_characters(
                raw_text
            ),
            "乱码检查_decode": "",
            "Token列表": ""
        }


    # ========================================================
    # 第一次 tokenizer：
    # 不截断
    #
    # 用于查看原始文字到底有多少 token
    # ========================================================

    full_encoding = tokenizer(
        cleaned_text,
        truncation=False,
        add_special_tokens=True,
        return_attention_mask=True
    )

    full_ids = full_encoding[
        "input_ids"
    ]

    full_token_count = len(
        full_ids
    )


    # ========================================================
    # 第二次 tokenizer：
    #
    # 参数与正式 FG-CLIP2 特征提取代码完全一致
    # ========================================================

    encoding = tokenizer(
        cleaned_text,

        padding="max_length",

        max_length=TEXT_MAX_LENGTH,

        truncation=True,

        return_attention_mask=True,

        return_tensors="pt"
    )


    input_ids = encoding[
        "input_ids"
    ][0]

    attention_mask = encoding[
        "attention_mask"
    ][0]


    # ========================================================
    # 只留下真正有效 token
    #
    # 去掉 padding
    # ========================================================

    valid_ids = input_ids[
        attention_mask.bool()
    ]


    actual_token_count = int(
        attention_mask.sum().item()
    )


    # ========================================================
    # ★ 最重要：
    #
    # 把模型真正收到的 token
    # 再 decode 回正常文字
    #
    # 这就是模型实际上看到的文本内容
    # ========================================================

    decoded_text = tokenizer.decode(
        valid_ids.tolist(),

        skip_special_tokens=True,

        clean_up_tokenization_spaces=False
    )


    # ========================================================
    # 查看 tokenizer 实际 token
    # ========================================================

    tokens = tokenizer.convert_ids_to_tokens(
        valid_ids.tolist()
    )

    token_string = " | ".join(
        tokens
    )


    # ========================================================
    # 是否截断
    # ========================================================

    is_truncated = (
        full_token_count
        >
        TEXT_MAX_LENGTH
    )


    # ========================================================
    # 简单一致性检查
    #
    # Tokenizer decode 可能会对空格产生正常的小调整，
    # 所以先归一化后比较。
    # ========================================================

    clean_compare = normalize_for_compare(
        cleaned_text
    )

    decode_compare = normalize_for_compare(
        decoded_text
    )


    if not is_truncated:

        if clean_compare == decode_compare:

            consistency = (
                "完全一致"
            )

        elif (
            clean_compare in decode_compare
            or decode_compare in clean_compare
        ):

            consistency = (
                "基本一致（可能存在 tokenizer 空格差异）"
            )

        else:

            consistency = (
                "需要人工检查"
            )

    else:

        consistency = (
            "文本被截断，请比较 decode 的前半部分"
        )


    return {

        "id":
            sample_id,

        "类型":
            text_type,

        "Excel原始文字":
            str(raw_text),

        "clean_text后文字":
            cleaned_text,

        "模型实际decode文字":
            decoded_text,

        "截断前token数量":
            full_token_count,

        "实际输入token数量":
            actual_token_count,

        "是否超过196":
            full_token_count > TEXT_MAX_LENGTH,

        "是否被截断":
            is_truncated,

        "文字一致性":
            consistency,

        "乱码检查_原文":
            detect_suspicious_characters(
                raw_text
            ),

        "乱码检查_decode":
            detect_suspicious_characters(
                decoded_text
            ),

        "Token列表":
            token_string
    }


# ============================================================
# 开始检查
# ============================================================

results = []


for _, row in sample_df.iterrows():

    sample_id = str(
        row.iloc[0]
    ).strip()

    raw_text = row.iloc[3]

    raw_description = row.iloc[4]


    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    result_t = inspect_text(
        sample_id,
        "T_text",
        raw_text
    )

    results.append(
        result_t
    )


    # --------------------------------------------------------
    # Description
    # --------------------------------------------------------

    result_c = inspect_text(
        sample_id,
        "C_description",
        raw_description
    )

    results.append(
        result_c
    )


# ============================================================
# 转成 DataFrame
# ============================================================

result_df = pd.DataFrame(
    results
)


# ============================================================
# 保存 Excel
# ============================================================

result_df.to_excel(
    OUTPUT_XLSX,
    index=False
)


# ============================================================
# 同时保存 UTF-8 TXT
#
# 避免 Windows 控制台编码影响判断
# ============================================================

with open(
    OUTPUT_TXT,
    "w",
    encoding="utf-8"
) as f:

    for item in results:

        f.write(
            "=" * 100
            + "\n"
        )

        f.write(
            f"ID：{item['id']}\n"
        )

        f.write(
            f"类型：{item['类型']}\n"
        )

        f.write(
            f"截断前 token 数："
            f"{item['截断前token数量']}\n"
        )

        f.write(
            f"实际输入 token 数："
            f"{item['实际输入token数量']}\n"
        )

        f.write(
            f"是否被截断："
            f"{item['是否被截断']}\n"
        )

        f.write(
            f"一致性："
            f"{item['文字一致性']}\n"
        )

        f.write(
            "\n【Excel 原始文字】\n"
        )

        f.write(
            str(
                item[
                    "Excel原始文字"
                ]
            )
            + "\n"
        )

        f.write(
            "\n【clean_text 后】\n"
        )

        f.write(
            str(
                item[
                    "clean_text后文字"
                ]
            )
            + "\n"
        )

        f.write(
            "\n【模型真正收到的 token decode 后文字】\n"
        )

        f.write(
            str(
                item[
                    "模型实际decode文字"
                ]
            )
            + "\n"
        )

        f.write(
            "\n【原文乱码检查】\n"
        )

        f.write(
            str(
                item[
                    "乱码检查_原文"
                ]
            )
            + "\n"
        )

        f.write(
            "\n【Decode 乱码检查】\n"
        )

        f.write(
            str(
                item[
                    "乱码检查_decode"
                ]
            )
            + "\n\n"
        )


# ============================================================
# 控制台显示
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "FG-CLIP2 文本输入检查结果"
)

print(
    "=" * 90
)


for item in results:

    print(
        f"\nID：{item['id']}"
    )

    print(
        f"类型：{item['类型']}"
    )

    print(
        f"截断前 token："
        f"{item['截断前token数量']}"
    )

    print(
        f"实际输入 token："
        f"{item['实际输入token数量']}"
    )

    print(
        f"是否被截断："
        f"{item['是否被截断']}"
    )

    print(
        f"一致性："
        f"{item['文字一致性']}"
    )

    print(
        "\nExcel 原始文字："
    )

    print(
        item[
            "Excel原始文字"
        ]
    )

    print(
        "\n模型实际收到的文字："
    )

    print(
        item[
            "模型实际decode文字"
        ]
    )

    print(
        "-" * 90
    )


print(
    "\n检查完成。"
)

print(
    f"\nExcel 检查结果：\n"
    f"{OUTPUT_XLSX}"
)

print(
    f"\nUTF-8 TXT 检查结果：\n"
    f"{OUTPUT_TXT}"
)