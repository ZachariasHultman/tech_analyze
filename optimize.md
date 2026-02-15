# Optimization Methods

## Overview

The system scores stocks on ~18 fundamental metrics, weights them, and sums into a total score. Optimization finds **weights** and **thresholds** (NOK/OK boundaries) that maximize the correlation between scores and actual forward stock returns.

Each metric has:
- **Weight** — how much it contributes to the total score (0 to 2.0)
- **Thresholds (NOK, OK)** — the boundaries that map a raw metric value to a score between -weight and +weight via linear interpolation

All three optimization methods now optimize both parameters simultaneously.

Three optimization methods are available, each trading off simplicity vs thoroughness.

---

## 1. `--optimize-individual` (Independent Correlation)

**How it works:**
1. For each metric independently, compute Spearman correlation with forward returns across TOTAL time windows (3Y, 5Y)
2. Drop metrics with correlation < 0.02
3. Scale remaining correlations to weights in [0, 2.0]
4. Apply constraints (momentum cap at 1.0, floors for Piotroski/dividend/earnings quality)
5. Sweep NOK/OK thresholds per-metric independently to find the pair that maximizes overall Spearman

**Strengths:**
- Simple and fast
- Robust against overfitting with small datasets
- Easy to interpret — each weight directly reflects that metric's predictive power

**Weakness:**
- Treats each metric independently — misses interactions
- Two weak metrics that are powerful together both get low weight
- Two redundant metrics both get high weight (double-counting)

**Output:** `optimization_results_individual.json`, `OPTIMIZED_THRESHOLDS_INDIVIDUAL` in metrics.py

---

## 2. `--optimize-combo` (Grid Sweep + Cross-Validation)

**How it works:**
1. Use the independent correlations from method 1 as a starting point (weights + thresholds)
2. Coordinate descent: for each metric, sweep both weight (±0.5 in 0.25 steps) AND thresholds (NOK/OK grid)
3. Use leave-one-out cross-validation across time windows to prevent overfitting
4. Repeat up to 3 rounds until converged
5. Apply the same constraints as method 1 (momentum cap, weight floors)

**Why this helps:**
- Tests actual combinations of weights AND thresholds, not just individual metrics
- Cross-validation prevents the optimizer from memorizing historical quirks
- The grid is seeded from method 1 results, so the search space stays manageable

**Trade-offs:**
- Slower than method 1
- Can still miss interactions if the grid is too coarse
- More complex to interpret

**Output:** `optimization_results_combo.json`, `OPTIMIZED_THRESHOLDS_COMBO` in metrics.py

---

## 3. `--optimize-stepwise` (Scipy Numerical Optimization)

**How it works:**
1. Start from the independent correlation weights + thresholds (method 1)
2. Use `scipy.optimize.minimize` (Nelder-Mead) to adjust all weights AND all thresholds simultaneously
3. Parameter vector: [weight₀, ..., weightₙ, nok₀, ok₀, ..., nokₙ, okₙ]
4. Objective: maximize average Spearman correlation across time windows (negated for minimization)
5. Constraints enforced via clipping: weights in [0, 2], threshold ordering (nok < ok for dir=+1), momentum cap, weight floors
6. Cross-validation same as method 2

**Why this helps:**
- Explores the full continuous weight + threshold space, not just grid points
- Can find subtle interactions that grid search misses
- Nelder-Mead is derivative-free, works well with noisy rank correlations

**Trade-offs:**
- May converge to local optima (mitigated by good starting point from method 1)
- Harder to interpret why specific weights/thresholds were chosen
- Risk of overfitting is higher — cross-validation is essential
- Larger parameter space (3 params per metric vs 1) needs more evaluations

**Output:** `optimization_results_stepwise.json`, `OPTIMIZED_THRESHOLDS_STEPWISE` in metrics.py

---

## What Gets Optimized

| Parameter        | Description                           | Example                          |
|------------------|---------------------------------------|----------------------------------|
| Weight           | How much a metric affects total score | `revenue y cagr status: 1.75`    |
| NOK threshold    | Below this → negative score           | `revenue y cagr: NOK=0.01`       |
| OK threshold     | Above this → positive score           | `revenue y cagr: OK=0.08`        |

The thresholds define the *shape* of scoring (where "good" starts), weights define the *magnitude*.

## Where Results Are Stored

- **Weights**: in `optimization_results_<method>.json`
- **Thresholds**: in both the JSON file AND as `OPTIMIZED_THRESHOLDS_<METHOD>` dicts in `metrics.py`
- Each method has its own separate dict, so they don't overwrite each other

## Comparison

| Aspect                  | `--optimize-individual` | `--optimize-combo`       | `--optimize-stepwise`     |
|-------------------------|-------------------------|--------------------------|---------------------------|
| Metric interactions     | No                      | Yes (grid)               | Yes (continuous)          |
| Threshold optimization  | Per-metric sweep        | Coordinate descent       | Simultaneous (Nelder-Mead)|
| Overfitting risk        | Low                     | Medium (CV mitigates)    | Higher (CV mitigates)     |
| Speed                   | Fast                    | Moderate                 | Slower                    |
| Interpretability        | High                    | Medium                   | Lower                     |
| Best for                | Small datasets          | Medium datasets          | Larger datasets           |

## Usage

```bash
# Independent correlation optimization
python analyzer/main.py --optimize-individual

# Grid sweep with cross-validation
python analyzer/main.py --optimize-combo

# Numerical optimization with cross-validation
python analyzer/main.py --optimize-stepwise

# Live analysis using specific optimization results (loads both weights + thresholds)
python analyzer/main.py                          # uses optimization_results_individual.json (default)
python analyzer/main.py --use-individual         # explicit: uses optimization_results_individual.json
python analyzer/main.py --use-combo              # uses optimization_results_combo.json
python analyzer/main.py --use-stepwise           # uses optimization_results_stepwise.json
python analyzer/main.py --no-opt                 # ignore all optimized weights and thresholds
```
