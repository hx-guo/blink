"""matplotlib 的中文字体：用思源黑体 / Noto Sans CJK SC（大陆规范字形）。

不要把 "PingFang SC" 放在首位：matplotlib 读不到苹方（系统私有字体），会静默退到
"Arial Unicode MS"，那个字体的汉字字形不是大陆规范写法。这里显式注册常见安装位置的
Noto CJK SC，找不到时退到冬青黑体（Hiragino Sans GB，也是简体规范字形）。
"""
import glob, os
import matplotlib.font_manager as fm

# 末尾的 DejaVu Sans 只补上标负号（⁻）等 CJK 字体里没有的符号；matplotlib ≥3.6 逐字形回退
FAMILIES = ["Noto Sans CJK SC", "Source Han Sans SC", "Hiragino Sans GB", "DejaVu Sans"]

for pat in ("~/Library/Fonts/NotoSansCJKsc-*.otf", "/Library/Fonts/NotoSansCJKsc-*.otf",
            "/usr/share/fonts/**/NotoSansCJK*.ttc", "~/Library/Fonts/SourceHanSansSC-*.otf"):
    for f in glob.glob(os.path.expanduser(pat), recursive=True):
        fm.fontManager.addfont(f)
