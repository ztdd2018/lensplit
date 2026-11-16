"""下载 Lensplit 台词识别所需的 whisper-small 模型（CTranslate2 格式）。

用法：
    python scripts/download_models.py            # 下载到 models/whisper-small
    python scripts/download_models.py --force    # 覆盖已存在的文件

说明：
    - 权重来自 HuggingFace 的 Systran/faster-whisper-small（openai/whisper-small 的 CT2 转换），
      约 484MB；仓库里只保留 config.json、tokenizer.json、vocabulary.txt 等小文件。
    - 只依赖标准库；下载完成后可直接启动 Lensplit，无需再联网。
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPOSITORY = "Systran/faster-whisper-small"
BASE_URL = f"https://huggingface.co/{REPOSITORY}/resolve/main"
FILES = ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")
TARGET_DIR = Path(__file__).resolve().parents[1] / "models" / "whisper-small"
CHUNK = 1024 * 1024


def download(name: str, force: bool) -> None:
    destination = TARGET_DIR / name
    if destination.is_file() and destination.stat().st_size > 0 and not force:
        print(f"跳过 {name}（已存在 {destination.stat().st_size / 1048576:.1f} MB）")
        return
    url = f"{BASE_URL}/{name}"
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"下载 {name} <- {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "lensplit-model-fetcher"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        received = 0
        while True:
            chunk = response.read(CHUNK)
            if not chunk:
                break
            handle.write(chunk)
            received += len(chunk)
            if total:
                print(f"\r  {received / 1048576:.1f} / {total / 1048576:.1f} MB", end="")
    print()
    temporary.replace(destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下载 whisper-small（CTranslate2）模型")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的文件")
    args = parser.parse_args(argv)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for name in FILES:
            download(name, args.force)
    except OSError as error:
        print(f"下载失败：{error}", file=sys.stderr)
        print("如果网络受限，可在能联网的机器上下载后，把文件放进 models/whisper-small/。", file=sys.stderr)
        return 1
    print(f"完成：{TARGET_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
