import argparse
import csv
import re
from pathlib import Path

from config import OUTPUT_DIR

RESULT_FILE = OUTPUT_DIR / "paper_reproduction_results.csv"


def clean_tag(tag: str) -> str:
    tag = tag.strip()
    if not tag:
        return ""
    return re.sub(r"[^0-9A-Za-z_\-]+", "_", tag)


def tagged_name(base: str, tag: str) -> str:
    tag = clean_tag(tag)
    if not tag:
        return base
    stem, suffix = base.rsplit(".", 1)
    return f"{stem}_{tag}.{suffix}"


def pct_to_float(value: str) -> float:
    return float(value.rstrip("%"))


def read_results(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Run run_experiments.py first: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(name: str, rows: list[dict], fieldnames: list[str], output_tag: str) -> None:
    out = OUTPUT_DIR / tagged_name(name, output_tag)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out}")


def make_table3(rows: list[dict], output_tag: str) -> None:
    seen = {}
    for row in rows:
        key = (row["排序组"], row["舆情场景"])
        if key not in seen:
            seen[key] = {
                "排序组": row["排序组"],
                "舆情场景": row["舆情场景"],
                "初始极化率": row["初始极化率"],
                "无干预最终极化率": row["无干预最终极化率"],
            }
    values = list(seen.values())
    write_csv("table3_baseline_polarization.csv", values, list(values[0].keys()), output_tag)


def make_best_table(rows: list[dict], max_p: float | None, out_name: str, output_tag: str) -> None:
    best = {}
    for row in rows:
        p = float(row["干预概率"])
        if max_p is not None and p > max_p:
            continue

        key = (row["排序组"], row["舆情场景"])
        candidate = (pct_to_float(row["极化率降低量"]), pct_to_float(row["干预接受率"]))
        if key not in best or candidate > best[key][0]:
            best[key] = (candidate, row)

    fieldnames = ["排序组", "舆情场景", "方法", "参数名", "参数值", "干预概率", "极化率降低量", "干预接受率", "干预后最终极化率"]
    out_rows = [{key: row[key] for key in fieldnames} for _, row in best.values()]
    write_csv(out_name, out_rows, fieldnames, output_tag)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", type=Path, default=RESULT_FILE)
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()

    rows = read_results(args.results_csv)
    make_table3(rows, args.output_tag)
    make_best_table(rows, None, "table4_best_full_intensity.csv", args.output_tag)
    make_best_table(rows, 0.3, "table5_best_low_intensity.csv", args.output_tag)


if __name__ == "__main__":
    main()
