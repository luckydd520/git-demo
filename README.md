# Weighted Random Exposure Reproduction

This folder contains a separate version of the paper experiment code using weighted random comment exposure.

The original uploaded GitHub folder is not modified. In this version, each agent randomly samples one comment from the global stream at each time step. Sampling probability is determined by the comment position within each topic-level Top 100 list.

## Exposure Rule

For a comment ranked `r` within a topic-level Top 100 list:

```text
w_r = 1 / r^alpha
```

The current setting is:

```text
alpha = 1.0
```

Thus, higher-ranked comments have higher exposure probability. The weights are normalized over the global stream before random sampling.

## Commands

Run a quick test:

```bash
python run_experiments.py --quick
```

Run the full experiment:

```bash
python run_experiments.py
```

Generate Tables 3, 4, and 5:

```bash
python make_tables.py
```
