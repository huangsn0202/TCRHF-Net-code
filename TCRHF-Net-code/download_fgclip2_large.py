# download_fgclip2_large.py
# 功能：
# 1. 将 Hugging Face 上已经预训练好的 FG-CLIP2-Large 下载到当前代码同目录
# 2. 保存到 ./FG-CLIP2-Large 文件夹
# 3. 后续可以完全从本地加载，不再联网下载模型
#
# 安装依赖：
# pip install -U huggingface_hub transformers safetensors sentencepiece

from pathlib import Path
from huggingface_hub import snapshot_download


# ============================================================
# 配置
# ============================================================

# FG-CLIP2-Large 官方 Hugging Face 模型
MODEL_ID = "qihoo360/fg-clip2-large"

# 当前代码所在目录
BASE_DIR = Path(__file__).resolve().parent

# 模型保存目录
MODEL_DIR = BASE_DIR / "FG-CLIP2-Large"


# ============================================================
# 下载模型
# ============================================================

def download_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("开始下载 FG-CLIP2-Large")
    print(f"模型来源：{MODEL_ID}")
    print(f"保存位置：{MODEL_DIR}")
    print("=" * 70)

    snapshot_download(
        repo_id=MODEL_ID,

        # 直接保存到指定文件夹
        local_dir=str(MODEL_DIR),

        # 下载完成后，整个模型都保存在本地目录
        local_dir_use_symlinks=False,

        # 如果下载过程中断，再次运行时会自动利用已经下载的文件
        resume_download=True,
    )

    print("\n" + "=" * 70)
    print("FG-CLIP2-Large 下载完成！")
    print(f"模型位置：{MODEL_DIR}")
    print("=" * 70)

    print("\n当前模型文件：")
    for file_path in sorted(MODEL_DIR.rglob("*")):
        if file_path.is_file():
            size_mb = file_path.stat().st_size / 1024 / 1024
            print(f"{file_path.relative_to(MODEL_DIR)}  ({size_mb:.2f} MB)")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    download_model()