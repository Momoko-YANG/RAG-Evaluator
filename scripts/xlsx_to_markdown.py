"""Convert Excel workbooks to Markdown for chat apps whose upload does not read .xlsx.

Many chat-upload pipelines read PDF / images (OCR) and .txt / .md, but not .xlsx.
This script writes one ``<name>.md`` per workbook: a ``## <sheet>`` section per
worksheet and one line per non-empty row, cells separated by `` | ``.  Date cells
are written as ``YYYY/MM/DD``.  The source workbooks are only read.

Usage
-----
    python scripts/xlsx_to_markdown.py datasets/my-project/source/*.xlsx \\
        --output-dir datasets/my-project/upload
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
_PKG_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}Relationship"

# Built-in number formats that are dates (ECMA-376 18.8.30).
_BUILTIN_DATE_FORMATS = {14, 15, 16, 17, 22, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                         50, 51, 52, 53, 54, 55, 56, 57, 58}
_EXCEL_EPOCH = datetime(1899, 12, 30)


def _text(el: ET.Element) -> str:
    """Text of a shared / inline string, without phonetic runs (<rPh>)."""
    parts = []
    for child in el:
        if child.tag == f"{_NS}t":
            parts.append(child.text or "")
        elif child.tag == f"{_NS}r":
            parts.extend(t.text or "" for t in child.iter(f"{_NS}t"))
    return "".join(parts)


def _column(ref: str) -> int:
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group():
        n = n * 26 + ord(ch) - 64
    return n - 1


def _date_styles(z: zipfile.ZipFile) -> set[int]:
    """Indexes of the cell styles (``s`` attribute) that format a date."""
    if "xl/styles.xml" not in z.namelist():
        return set()
    root = ET.fromstring(z.read("xl/styles.xml"))
    custom = set()
    for fmt in root.iter(f"{_NS}numFmt"):
        code = re.sub(r'"[^"]*"|\[[^\]]*\]', "", fmt.get("formatCode", "")).lower()
        if re.search(r"[ymd]", code) and "h" not in code:
            custom.add(int(fmt.get("numFmtId")))
    xfs = root.find(f"{_NS}cellXfs")
    if xfs is None:
        return set()
    return {
        i for i, xf in enumerate(xfs)
        if int(xf.get("numFmtId", 0)) in _BUILTIN_DATE_FORMATS | custom
    }


def _cell_value(cell: ET.Element, shared: list[str], date_styles: set[int]) -> str:
    kind = cell.get("t")
    if kind == "inlineStr":
        node = cell.find(f"{_NS}is")
        return _text(node) if node is not None else ""
    v = cell.find(f"{_NS}v")
    if v is None or v.text is None:
        return ""
    if kind == "s":
        return shared[int(v.text)]
    if kind in (None, "n") and int(cell.get("s", 0)) in date_styles:
        try:
            return (_EXCEL_EPOCH + timedelta(days=float(v.text))).strftime("%Y/%m/%d")
        except (ValueError, OverflowError):
            return v.text
    return v.text


def read_workbook(path: Path) -> list[tuple[str, list[list[str]]]]:
    """Return ``[(sheet name, rows)]`` with each row's non-empty cell values in order."""
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            shared = [_text(si) for si in root.findall(f"{_NS}si")]
        date_styles = _date_styles(z)
        workbook = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        targets = {r.get("Id"): r.get("Target") for r in rels.iter(_PKG_REL)}

        sheets = []
        for sheet in workbook.find(f"{_NS}sheets"):
            target = targets[sheet.get(_REL_ID)].lstrip("/")
            target = target if target.startswith("xl/") else f"xl/{target}"
            root = ET.fromstring(z.read(target))
            rows = []
            for row in root.iter(f"{_NS}row"):
                cells = sorted(
                    (_column(c.get("r")), _cell_value(c, shared, date_styles))
                    for c in row.findall(f"{_NS}c")
                )
                values = [" ".join(v.split()) for _, v in cells]
                values = [v for v in values if v]
                if values:
                    rows.append(values)
            sheets.append((sheet.get("name"), rows))
    return sheets


def to_markdown(title: str, sheets: list[tuple[str, list[list[str]]]]) -> str:
    lines = [f"# {title}", ""]
    for name, rows in sheets:
        lines += [f"## {name}", ""]
        lines += [" | ".join(row) for row in rows]
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("workbooks", nargs="+", type=Path, help=".xlsx files to convert.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in args.workbooks:
        if path.suffix.lower() != ".xlsx":
            logger.error("Not an .xlsx file: %s", path)
            sys.exit(1)
        sheets = read_workbook(path)
        out = args.output_dir / f"{path.stem}.md"
        out.write_text(to_markdown(path.stem, sheets), encoding="utf-8", newline="\n")
        logger.info("%s -> %s (%d sheets)", path.name, out, len(sheets))


if __name__ == "__main__":
    main()
