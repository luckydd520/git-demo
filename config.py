from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RAW_DATA_ROOT = ROOT.parent / "手动情感分析"
SENTIMENT_SCORE_FILE = ROOT / 'data' / 'sentiment_scores_16files.json'
DATA_SOURCE = 'sentiment_json_continuous'

SENTIMENT_MAPPING = {1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0}

EXPECTED_PAPER_TOTAL = 5604
TOP_N = 100
NUM_AGENTS = 50
LEARNING_RATE = 0.3
TRUST_LOW = 0.0
TRUST_HIGH = 0.5
POLARIZATION_THRESHOLD = 0.8
REPETITIONS = 10

# Weighted random exposure: weight for rank r is 1 / r ** EXPOSURE_ALPHA.
EXPOSURE_ALPHA = 1.0
BASE_SEED = 42

PROBS = [round(i * 0.1, 1) for i in range(1, 10)]
SCARCITY_K = [10, 20, 30]
SUPPRESSION_GAMMA = [0.5, 1.0, 2.0]

SCENARIOS = {
    "polarization": {"label": "激烈骂战(U型)", "alpha": 0.5, "beta": 0.5},
    "fan_circle": {"label": "饭圈控评(J型)", "alpha": 5.0, "beta": 0.5},
    "blackout": {"label": "全网黑(反J型)", "alpha": 0.5, "beta": 5.0},
    "positive_bias": {"label": "普遍好评", "alpha": 4.0, "beta": 2.0},
    "negative_bias": {"label": "普遍差评", "alpha": 2.0, "beta": 4.0},
    "consensus": {"label": "理性讨论(钟型)", "alpha": 2.0, "beta": 2.0},
    "random": {"label": "无序状态", "alpha": 1.0, "beta": 1.0},
}

RANK_GROUPS = {
    "HotRank": "热度排序.xlsx",
    "TimeRank": "时间排序.xlsx",
}

