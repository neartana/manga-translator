#!/usr/bin/env python3
"""Fetch the starter font packs (open-license fonts).

The GitHub repo ships without the binary font files to keep it light.
Run this once after cloning:

    python scripts/download_fonts.py

Fonts downloaded (both SIL Open Font License 1.1):
  - Comic Neue (regular + bold)      -> fonts/en/
  - Noto Sans CJK (regular, TTC)     -> fonts/default/
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FONTS = ROOT / "fonts"

COMIC_NEUE = (
    "https://fonts.google.com/download?family=Comic%20Neue",  # zip archive
    "en",
    ("ComicNeue-Regular.ttf", "ComicNeue-Bold.ttf"),
)
NOTO_CJK = (
    "https://github.com/notofonts/noto-cjk/raw/main/Sans/SubsetOTF/TC/NotoSansTC-Regular.otf",
    "default",
    None,
)


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=120).read()


def main() -> int:
    (FONTS / "en").mkdir(parents=True, exist_ok=True)
    (FONTS / "default").mkdir(parents=True, exist_ok=True)

    print("Downloading Comic Neue…")
    try:
        data = fetch(COMIC_NEUE[0])
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        with zipfile.ZipFile(tmp_path) as zf:
            for name in COMIC_NEUE[2]:
                for member in zf.namelist():
                    if member.endswith(name):
                        (FONTS / "en" / name).write_bytes(zf.read(member))
                        print(f"  fonts/en/{name}")
                        break
    except Exception as e:
        print(f"  Comic Neue failed: {e}")

    print("Downloading Noto Sans CJK…")
    try:
        data = fetch(NOTO_CJK[0])
        (FONTS / "default" / "NotoSansCJK-Regular.otf").write_bytes(data)
        print("  fonts/default/NotoSansCJK-Regular.otf")
    except Exception as e:
        print(f"  Noto CJK failed: {e}")
        print("  (any CJK .ttf/.otf dropped into fonts/default/ works too)")

    print("Done. Add your own packs under fonts/<lang>/ — see fonts/README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
