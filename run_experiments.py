import argparse
import csv
import re
from copy import deepcopy
from pathlib import Path

from config import BASE_SEED, OUTPUT_DIR, PROBS, REPETITIONS, SCARCITY_K, SCENARIOS, SENTIMENT_SCORE_FILE, SUPPRESSION_GAMMA
from data_loader import load_rank_groups
from interventions import baseline_stream, intervention_stream
from metrics import average_metrics, summarize_run
from model import make_agents, run_model


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


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


def experiment_rows(quick: bool = False, score_file: Path | None = None, manifest_name: str = "data_manifest.csv") -> list[dict]:
    groups, _ = load_rank_groups(score_file=score_file, manifest_name=manifest_name)
    rows = []

    scenarios = dict(list(SCENARIOS.items())[:1]) if quick else SCENARIOS
    probs = [0.2] if quick else PROBS
    methods = ({"scarcity": SCARCITY_K[:1], "suppression": SUPPRESSION_GAMMA[:1]}
               if quick else {"scarcity": SCARCITY_K, "suppression": SUPPRESSION_GAMMA})
    repetitions = 2 if quick else REPETITIONS

    for group_name, files in groups.items():
        if not files:
            continue
        base_stream = baseline_stream(files)

        for scenario_name, scenario in scenarios.items():
            alpha = scenario["alpha"]
            beta = scenario["beta"]

            base_runs = []
            for rep in range(repetitions):
                seed = BASE_SEED + rep
                agents_initial = make_agents(alpha, beta, seed)
                agents_no = deepcopy(agents_initial)
                no_result = run_model(base_stream, agents_no, seed=seed)
                base_runs.append((seed, agents_initial, no_result))

            for method, params in methods.items():
                for param in params:
                    for p in probs:
                        rep_metrics = []
                        with_stream = intervention_stream(files, method, p, param)

                        for seed, agents_initial, no_result in base_runs:
                            agents_with = deepcopy(agents_initial)
                            with_result = run_model(with_stream, agents_with, seed=seed)
                            rep_metrics.append(summarize_run(no_result, with_result))

                        avg = average_metrics(rep_metrics)
                        rows.append({
                            "排序组": group_name,
                            "舆情场景": scenario["label"],
                            "场景代码": scenario_name,
                            "方法": method,
                            "参数名": "K" if method == "scarcity" else "Gamma",
                            "参数值": param,
                            "干预概率": p,
                            "重复次数": repetitions,
                            "初始极化率": percent(avg["pol_init"]),
                            "无干预最终极化率": percent(avg["pol_no"]),
                            "干预后最终极化率": percent(avg["pol_with"]),
                            "极化率降低量": percent(avg["pol_reduction"]),
                            "观点均值_初始": f"{avg['mean_init']:.4f}",
                            "观点均值_无干预": f"{avg['mean_no']:.4f}",
                            "观点均值_干预后": f"{avg['mean_with']:.4f}",
                            "观点标准差_初始": f"{avg['std_init']:.4f}",
                            "观点标准差_无干预": f"{avg['std_no']:.4f}",
                            "观点标准差_干预后": f"{avg['std_with']:.4f}",
                            "干预接受率": percent(avg["acceptance_rate"]),
                        })

    return rows


def write_rows(rows: list[dict], quick: bool, output_tag: str) -> Path:
    base = "quick_results.csv" if quick else "paper_reproduction_results.csv"
    out = OUTPUT_DIR / tagged_name(base, output_tag)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a tiny smoke test.")
    parser.add_argument("--score-file", type=Path, default=SENTIMENT_SCORE_FILE, help="Continuous score JSON to use.")
    parser.add_argument("--output-tag", default="", help="Suffix for output files, for example rebuilt.")
    args = parser.parse_args()

    manifest_name = tagged_name("data_manifest.csv", args.output_tag)
    rows = experiment_rows(quick=args.quick, score_file=args.score_file, manifest_name=manifest_name)
    if not rows:
        raise RuntimeError(f"No experiment rows were produced. Check {manifest_name}.")
    write_rows(rows, quick=args.quick, output_tag=args.output_tag)


if __name__ == "__main__":
    main()

