# -*- coding: utf-8 -*-
"""提取 HTML 中所有内联 <script> 块，拼接到一个 JS 文件（供 node --check 语法检查）。

用法：python extract_js.py <index.html> <out.js>
"""
import re
import sys
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit("用法: python extract_js.py <index.html> <out.js>")

html = Path(sys.argv[1]).read_text(encoding="utf-8-sig")
blocks = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, re.S)
Path(sys.argv[2]).write_text("\n;\n".join(blocks), encoding="utf-8")
print(f"提取 {len(blocks)} 个内联 script 块 -> {sys.argv[2]}")
