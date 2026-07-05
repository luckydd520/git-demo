import numpy as np

from config import EXPOSURE_ALPHA, TOP_N
from data_loader import Comment


def base_score(n: int) -> np.ndarray:
    if n <= 1:
        return np.array([1.0])
    return np.linspace(1.0, 0.0, n)


def exposure_weight(rank_zero_based: int) -> float:
    rank = rank_zero_based + 1
    return 1.0 / (rank ** EXPOSURE_ALPHA)


def scarcity_score(comments: list[Comment], reference_pool: list[Comment], k: int) -> np.ndarray:
    values = np.array([c.value for c in comments], dtype=float)
    ref_values = np.array([c.value for c in reference_pool], dtype=float)
    bins = np.linspace(0.0, 1.0, k + 1)

    ref_idx = np.digitize(ref_values, bins, right=False)
    ref_idx = np.clip(ref_idx, 1, k)
    counts = np.bincount(ref_idx, minlength=k + 1)
    probs = counts / max(1, len(reference_pool))

    candidate_idx = np.digitize(values, bins, right=False)
    candidate_idx = np.clip(candidate_idx, 1, k)
    return 1.0 - probs[candidate_idx]


def suppression_score(comments: list[Comment], gamma: float) -> np.ndarray:
    values = np.array([c.value for c in comments], dtype=float)
    extremity = 2.0 * np.abs(values - 0.5)
    return np.power(1.0 - extremity, gamma)


def rerank(comments: list[Comment], method: str, p: float, param: float) -> list[tuple[Comment, bool, float]]:
    n = len(comments)
    if n == 0:
        return []

    s_base = base_score(n)
    if method == "scarcity":
        reference_pool = comments[:TOP_N]
        s_div = scarcity_score(comments, reference_pool, int(param))
    elif method == "suppression":
        s_div = suppression_score(comments, float(param))
    else:
        raise ValueError(f"Unknown method: {method}")

    s_final = (1.0 - p) * s_base + p * s_div
    ranked = sorted(zip(s_final, comments), key=lambda x: x[0], reverse=True)

    top = []
    for new_rank, (_, comment) in enumerate(ranked[:TOP_N]):
        is_intervened = comment.original_rank != new_rank
        top.append((comment, is_intervened, exposure_weight(new_rank)))
    return top


def baseline_stream(files: list[list[Comment]]) -> list[tuple[float, bool, float]]:
    stream = []
    for comments in files:
        for rank, comment in enumerate(comments[:TOP_N]):
            stream.append((comment.value, False, exposure_weight(rank)))
    return stream


def intervention_stream(files: list[list[Comment]], method: str, p: float, param: float) -> list[tuple[float, bool, float]]:
    stream = []
    for comments in files:
        for comment, is_intervened, weight in rerank(comments, method, p, param):
            stream.append((comment.value, is_intervened, weight))
    return stream
