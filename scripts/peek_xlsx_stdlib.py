"""Peek first rows of xlsx without external deps."""
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def load_shared_strings(z: zipfile.ZipFile) -> list[str]:
    shared: list[str] = []
    if "xl/sharedStrings.xml" not in z.namelist():
        return shared
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    for si in root.findall(f"{NS}si"):
        parts: list[str] = []
        t = si.find(f"{NS}t")
        if t is not None and t.text:
            parts.append(t.text)
        else:
            for r in si.findall(f".//{NS}t"):
                parts.append(r.text or "")
        shared.append("".join(parts))
    return shared


def cell_ref(ref: str) -> tuple[str, int]:
    m = re.match(r"([A-Z]+)(\d+)", ref, re.I)
    if not m:
        raise ValueError(ref)
    return m.group(1).upper(), int(m.group(2))


def read_sheet(path: Path, sheet_rel: str = "xl/worksheets/sheet1.xml", max_row: int = 40) -> None:
    with zipfile.ZipFile(path) as z:
        shared = load_shared_strings(z)
        root = ET.fromstring(z.read(sheet_rel))
        rows: dict[int, dict[str, object]] = {}
        for row in root.findall(f".//{NS}row"):
            for c in row.findall(f"{NS}c"):
                ref = c.attrib.get("r")
                if not ref:
                    continue
                col_letters, row_num = cell_ref(ref)
                if row_num > max_row:
                    continue
                v = c.find(f"{NS}v")
                if v is None or v.text is None:
                    val: object = None
                else:
                    raw = v.text
                    if c.attrib.get("t") == "s":
                        val = shared[int(raw)]
                    else:
                        try:
                            f = float(raw)
                            val = int(f) if f == int(f) else f
                        except ValueError:
                            val = raw
                rows.setdefault(row_num, {})[col_letters] = val
        def col_sort_key(col: str) -> tuple[int, str]:
            n = 0
            for ch in col:
                n = n * 26 + (ord(ch) - ord("A") + 1)
            return (n, col)

        for r in sorted(rows):
            cols = rows[r]
            keys = sorted(cols.keys(), key=col_sort_key)
            print(r, [cols[k] for k in keys])


if __name__ == "__main__":
    read_sheet(Path(__file__).resolve().parents[1] / "Forecasting Case- Study.xlsx")
