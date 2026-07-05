from dataclasses import dataclass

import numpy as np

from config import LEARNING_RATE, NUM_AGENTS, TRUST_HIGH, TRUST_LOW


@dataclass
class Agent:
    agent_id: int
    opinion: float
    tolerance: float
    intervened_seen: int = 0
    intervened_accepted: int = 0

    def read(self, value: float, is_intervened: bool) -> None:
        if is_intervened:
            self.intervened_seen += 1

        if abs(value - self.opinion) <= self.tolerance:
            self.opinion += (value - self.opinion) * LEARNING_RATE
            self.opinion = min(1.0, max(0.0, self.opinion))
            if is_intervened:
                self.intervened_accepted += 1


def make_agents(alpha: float, beta: float, seed: int) -> list[Agent]:
    rng = np.random.default_rng(seed)
    opinions = rng.beta(alpha, beta, NUM_AGENTS)
    tolerances = rng.uniform(TRUST_LOW, TRUST_HIGH, NUM_AGENTS)
    return [Agent(i, float(opinions[i]), float(tolerances[i])) for i in range(NUM_AGENTS)]


def run_model(stream: list[tuple[float, bool, float]], agents: list[Agent], seed: int) -> dict:
    initial = np.array([a.opinion for a in agents], dtype=float)
    if not stream:
        return {"initial": initial, "final": initial.copy(), "acceptance_rate": 0.0}

    values = np.array([item[0] for item in stream], dtype=float)
    intervened = np.array([item[1] for item in stream], dtype=bool)
    weights = np.array([item[2] for item in stream], dtype=float)
    probabilities = weights / weights.sum()
    cumulative_probabilities = np.cumsum(probabilities)
    cumulative_probabilities[-1] = 1.0

    opinions = initial.copy()
    tolerances = np.array([a.tolerance for a in agents], dtype=float)
    intervened_seen = np.zeros(len(agents), dtype=int)
    intervened_accepted = np.zeros(len(agents), dtype=int)

    rng = np.random.default_rng(seed)
    steps = len(stream)
    for _ in range(steps):
        sampled_indices = np.searchsorted(cumulative_probabilities, rng.random(len(agents)), side="right")
        sampled_values = values[sampled_indices]
        sampled_intervened = intervened[sampled_indices]
        accepted = np.abs(sampled_values - opinions) <= tolerances

        intervened_seen += sampled_intervened.astype(int)
        intervened_accepted += (sampled_intervened & accepted).astype(int)
        opinions[accepted] += (sampled_values[accepted] - opinions[accepted]) * LEARNING_RATE
        opinions = np.clip(opinions, 0.0, 1.0)

    for i, agent in enumerate(agents):
        agent.opinion = float(opinions[i])
        agent.intervened_seen = int(intervened_seen[i])
        agent.intervened_accepted = int(intervened_accepted[i])

    final = opinions
    seen = int(intervened_seen.sum())
    accepted_count = int(intervened_accepted.sum())
    acceptance_rate = accepted_count / seen if seen else 0.0

    return {"initial": initial, "final": final, "acceptance_rate": acceptance_rate}
