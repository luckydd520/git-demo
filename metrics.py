import numpy as np

from config import POLARIZATION_THRESHOLD


def polarization_rate(opinions: np.ndarray) -> float:
    low = 1.0 - POLARIZATION_THRESHOLD
    high = POLARIZATION_THRESHOLD
    return float(np.mean((opinions <= low) | (opinions >= high)))


def summarize_run(no_result: dict, with_result: dict) -> dict:
    pol_init = polarization_rate(no_result["initial"])
    pol_no = polarization_rate(no_result["final"])
    pol_with = polarization_rate(with_result["final"])
    return {
        "pol_init": pol_init,
        "pol_no": pol_no,
        "pol_with": pol_with,
        "pol_reduction": pol_no - pol_with,
        "mean_init": float(np.mean(no_result["initial"])),
        "mean_no": float(np.mean(no_result["final"])),
        "mean_with": float(np.mean(with_result["final"])),
        "std_init": float(np.std(no_result["initial"])),
        "std_no": float(np.std(no_result["final"])),
        "std_with": float(np.std(with_result["final"])),
        "acceptance_rate": with_result["acceptance_rate"],
    }


def average_metrics(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}
