"""Tests for scripts/xlsx_to_markdown.py."""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "xlsx_to_markdown.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("xlsx_to_markdown", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


converter = _load_module()

_MAIN = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
_REL = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'


def _write_xlsx(path: Path) -> None:
    """A two-sheet workbook: shared string with furigana, a date, a number, a blank row."""
    files = {
        "xl/workbook.xml": (
            f'<workbook {_MAIN} {_REL}><sheets>'
            '<sheet name="規程" sheetId="1" r:id="rId1"/>'
            '<sheet name="空" sheetId="2" r:id="rId2"/>'
            "</sheets></workbook>"
        ),
        "xl/_rels/workbook.xml.rels": (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/>'
            '<Relationship Id="rId2" Target="/xl/worksheets/sheet2.xml"/>'
            "</Relationships>"
        ),
        "xl/sharedStrings.xml": (
            f"<sst {_MAIN}>"
            "<si><t>承認</t><rPh sb=\"0\" eb=\"2\"><t>ショウニン</t></rPh></si>"
            "<si><t>改行\nあり</t></si>"
            "</sst>"
        ),
        "xl/styles.xml": (
            f'<styleSheet {_MAIN}><cellXfs count="2">'
            '<xf numFmtId="0"/><xf numFmtId="14"/>'
            "</cellXfs></styleSheet>"
        ),
        "xl/worksheets/sheet1.xml": (
            f"<worksheet {_MAIN}><sheetData>"
            '<row r="1"><c r="B1" t="s"><v>0</v></c><c r="A1" t="s"><v>1</v></c></row>'
            '<row r="2"><c r="A2"/></row>'
            '<row r="3"><c r="A3" s="1"><v>45463</v></c><c r="C3"><v>12.5</v></c></row>'
            "</sheetData></worksheet>"
        ),
        "xl/worksheets/sheet2.xml": f"<worksheet {_MAIN}><sheetData/></worksheet>",
    }
    with zipfile.ZipFile(path, "w") as z:
        for name, xml in files.items():
            z.writestr(name, xml)


def test_read_workbook(tmp_path: Path) -> None:
    path = tmp_path / "book.xlsx"
    _write_xlsx(path)
    assert converter.read_workbook(path) == [
        ("規程", [["改行 あり", "承認"], ["2024/06/20", "12.5"]]),
        ("空", []),
    ]


def test_main_writes_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "規程集.xlsx"
    _write_xlsx(source)
    before = source.read_bytes()
    out = tmp_path / "upload"
    monkeypatch.setattr(sys, "argv", ["xlsx_to_markdown.py", str(source), "--output-dir", str(out)])
    converter.main()

    md = (out / "規程集.md").read_text(encoding="utf-8")
    assert md.startswith("# 規程集\n\n## 規程\n\n改行 あり | 承認\n2024/06/20 | 12.5\n")
    assert "## 空" in md
    assert source.read_bytes() == before
