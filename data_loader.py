import csv
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from config import (
    DATA_SOURCE,
    EXPECTED_PAPER_TOTAL,
    OUTPUT_DIR,
    RANK_GROUPS,
    RAW_DATA_ROOT,
    SENTIMENT_MAPPING,
    SENTIMENT_SCORE_FILE,
)


@dataclass(frozen=True)
class Comment:
    value: float
    source_file: str
    original_rank: int


def load_rank_groups(score_file: Path | None = None, manifest_name: str = "data_manifest.csv") -> tuple[dict[str, list[list[Comment]]], list[dict]]:
    if DATA_SOURCE == "raw_excel_discrete":
        return load_raw_excel_groups(manifest_name=manifest_name)
    if DATA_SOURCE == "sentiment_json_continuous":
        return load_json_groups(score_file=score_file, manifest_name=manifest_name)
    raise ValueError(f"Unknown DATA_SOURCE: {DATA_SOURCE}")


def load_json_groups(score_file: Path | None = None, manifest_name: str = "data_manifest.csv") -> tuple[dict[str, list[list[Comment]]], list[dict]]:
    raw = load_sentiment_json(score_file)
    groups: dict[str, list[list[Comment]]] = {name: [] for name in RANK_GROUPS}
    manifest = []

    for file_name, items in raw.items():
        matched_group = group_for_file(file_name)
        count = len(items)
        manifest.append({"file_name": file_name, "rank_group": matched_group or "unused", "count": count})
        if matched_group:
            groups[matched_group].append(_extract_json_scores(file_name, items))

    finalize_manifest(manifest, manifest_name=manifest_name)
    return groups, manifest


def load_raw_excel_groups(manifest_name: str = "data_manifest.csv") -> tuple[dict[str, list[list[Comment]]], list[dict]]:
    if not RAW_DATA_ROOT.exists():
        raise FileNotFoundError(f"Raw data folder not found: {RAW_DATA_ROOT}")

    groups: dict[str, list[list[Comment]]] = {name: [] for name in RANK_GROUPS}
    manifest = []
    for path in sorted(RAW_DATA_ROOT.rglob("*.xlsx")):
        matched_group = group_for_file(path.name)
        comments = _extract_excel_scores(path)
        manifest.append(
            {
                "file_name": str(path.relative_to(RAW_DATA_ROOT)),
                "rank_group": matched_group or "unused",
                "count": len(comments),
            }
        )
        if matched_group:
            groups[matched_group].append(comments)

    finalize_manifest(manifest, manifest_name=manifest_name)
    return groups, manifest


def group_for_file(file_name: str) -> str | None:
    for group_name, suffix in RANK_GROUPS.items():
        if file_name.endswith(suffix):
            return group_name
    return None


def load_sentiment_json(path: Path | None = None) -> dict:
    path = path or SENTIMENT_SCORE_FILE
    if not path.exists():
        raise FileNotFoundError(f"Sentiment score file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_json_scores(file_name: str, items: dict) -> list[Comment]:
    comments = []
    for rank, item in enumerate(items.values()):
        comments.append(Comment(float(item["continuous_score"]), file_name, rank))
    return comments


def _extract_excel_scores(path: Path) -> list[Comment]:
    rows = _read_xlsx_rows(path)
    if not rows:
        return []

    headers = [str(x).strip() for x in rows[0]]
    try:
        sentiment_idx = headers.index("情感等级")
    except ValueError as exc:
        raise ValueError(f"No 情感等级 column in {path}") from exc

    comments = []
    for row in rows[1:]:
        if sentiment_idx >= len(row):
            continue
        level = _to_int(row[sentiment_idx])
        if level not in SENTIMENT_MAPPING:
            continue
        comments.append(Comment(SENTIMENT_MAPPING[level], path.name, len(comments)))
    return comments


def _to_int(value) -> int | None:
    if value is None or value == "":
        return None
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def _read_xlsx_rows(path: Path) -> list[list[str]]:
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("main:si", ns):
                texts = [node.text or "" for node in si.findall(".//main:t", ns)]
                shared.append("".join(texts))

        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet.findall(".//main:row", ns):
            values = []
            current_col = 0
            for cell in row.findall("main:c", ns):
                ref = cell.attrib.get("r", "")
                col_idx = _column_index(ref)
                while current_col < col_idx:
                    values.append("")
                    current_col += 1

                value_node = cell.find("main:v", ns)
                raw = value_node.text if value_node is not None else ""
                if cell.attrib.get("t") == "s" and raw != "":
                    value = shared[int(raw)]
                else:
                    value = raw
                values.append(value)
                current_col += 1
            rows.append(values)
    return rows


def _column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch.upper()) - ord("A") + 1)
    return max(0, value - 1)


def finalize_manifest(manifest: list[dict], manifest_name: str = "data_manifest.csv") -> None:
    used_total = sum(row["count"] for row in manifest if row["rank_group"] != "unused")
    manifest.append({"file_name": "__TOTAL_USED__", "rank_group": "all", "count": used_total})
    manifest.append({"file_name": "__EXPECTED_IN_PAPER__", "rank_group": "all", "count": EXPECTED_PAPER_TOTAL})
    write_manifest(manifest, manifest_name=manifest_name)


def write_manifest(manifest: list[dict], manifest_name: str = "data_manifest.csv") -> None:
    out = OUTPUT_DIR / manifest_name
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "rank_group", "count"])
        writer.writeheader()
        writer.writerows(manifest)
