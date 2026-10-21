#!/usr/bin/env python3
"""Extract reviewer annotations from a PDF (highlights, notes, strikeouts...).

For each annotation prints: page, type, the text span it covers (for
markup types), and the comment content. Used to round-trip paper
revisions: annotate main_en.pdf / main_zh.pdf in any PDF reader, then
run this to get an actionable list.

Handwritten (Apple Pencil / Ink) marks carry no text; pages containing
them are rendered to PNG under --render-dir for visual reading.

Usage: .venv/bin/python scripts/tools/extract_pdf_annots.py <annotated.pdf>
           [--render-dir DIR]
"""
import argparse
import os
import sys

import fitz  # PyMuPDF


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--render-dir", default="/tmp/pdf_annots")
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    n = 0
    ink_pages = set()
    for pno, page in enumerate(doc, 1):
        for an in page.annots() or []:
            kind = an.type[1]
            quoted = ""
            if an.vertices and kind in (
                "Highlight", "Underline", "StrikeOut", "Squiggly",
            ):
                quads = [
                    fitz.Quad(an.vertices[i : i + 4]).rect
                    for i in range(0, len(an.vertices), 4)
                ]
                quoted = " ".join(
                    page.get_textbox(r).strip() for r in quads
                ).strip()
            elif kind in ("FreeText", "Text", "Square", "Circle"):
                # note anchored to a region: grab nearby text for context
                quoted = page.get_textbox(an.rect + (-5, -5, 5, 5)).strip()
            elif kind in ("Ink", "Stamp", "Line", "Polygon", "PolyLine"):
                ink_pages.add(pno)
            content = (an.info.get("content") or "").strip()
            n += 1
            print(f"--- #{n}  p{pno}  [{kind}]")
            if quoted:
                print(f"    span: {quoted}")
            if content:
                print(f"    note: {content}")
    if n == 0:
        print("no annotations found (handwriting may be flattened; "
              "render pages manually and read them visually)")
    if ink_pages:
        os.makedirs(args.render_dir, exist_ok=True)
        print(f"\nhandwritten-ink pages rendered to {args.render_dir}:")
        for pno in sorted(ink_pages):
            pix = doc[pno - 1].get_pixmap(dpi=170, annots=True)
            out = os.path.join(args.render_dir, f"p{pno:03d}.png")
            pix.save(out)
            print(f"    {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
