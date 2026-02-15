# Optimization Methods

## Overview

The system scores stocks on ~18 fundamental metrics, weights them, and sums into a total score. Optimization finds weights that maximize the correlation between scores and actual forward stock returns.

Three optimization methods are available, each trading off simplicity vs thoroughness.

---

## 1. `--optimize-individual` (Independent Correlation)

**How it works:**
1. For each metric independently, compute Spearman correlation with forward returns across TOTAL time windows (3Y, 5Y)
2. Drop metrics with correlation < 0.02
3. Scale remaining correlations to weights in [0, 2.0]
4. Apply constraints (momentum cap at 1.0, floors for Piotroski/dividend/earnings quality)

**Strengths:**
- Simple and fast
- Robust against overfitting with small datasets
- Easy to interpret — each weight directly reflects that metric's predictive power

**Weakness:**
- Treats each metric independently — misses interactions
- Two weak metrics that are powerful together both get low weight
- Two redundant metrics both get high weight (double-counting)

**Output:** `optimization_results_individual.json`

---

## 2. `--optimize-combo` (Grid Sweep + Cross-Validation)

**How it works:**
1. Use the independent correlations from method 1 as a starting point
2. Define a weight grid around each metric's starting weight (e.g. ±0.5 in steps of 0.25)
3. For each weight combination, score all companies and compute Spearman correlation with returns
4. Use k-fold cross-validation (train on k-1 time windows, validate on the held-out window) to prevent overfitting
5. Select the weight combination with the best average validation correlation
6. Apply the same constraints as method 1 (momentum cap, weight floors)

**Why this helps:**
- Tests actual combinations of weights, not just individual metrics
- Cross-validation prevents the optimizer from memorizing historical quirks
- The grid is seeded from method 1 results, so the search space stays manageable

**Trade-offs:**
- Slower than method 1 (grid search is O(n^k) but bounded by narrow ranges)
- Can still miss interactions if the grid is too coarse
- More complex to interpret

**Output:** `optimization_results_combo.json`

---

## 3. `--optimize-stepwise` (Scipy Numerical Optimization)

**How it works:**
1. Start from the independent correlation weights (method 1)
2. Use `scipy.optimize.minimize` (Nelder-Mead) to adjust all weights simultaneously
3. Objective: maximize average Spearman correlation across time windows (negated for minimization)
4. Constraints enforced via clipping: weights in [0, 2], momentum cap, weight floors
5. Cross-validation same as method 2

**Why this helps:**
- Explores the full continuous weight space, not just grid points
- Can find subtle interactions that grid search misses
- Nelder-Mead is derivative-free, works well with noisy rank correlations

**Trade-offs:**
- May converge to local optima (mitigated by good starting point from method 1)
- Harder to interpret why specific weights were chosen
- Risk of overfitting is higher — cross-validation is essential

**Output:** `optimization_results_stepwise.json`

---

## Comparison

| Aspect                  | `--optimize-individual` | `--optimize-combo`       | `--optimize-stepwise`     |
|-------------------------|---------------------|--------------------------|---------------------------|
| Metric interactions     | No                  | Yes (grid)               | Yes (continuous)          |
| Overfitting risk        | Low                 | Medium (CV mitigates)    | Higher (CV mitigates)     |
| Speed                   | Fast                | Moderate                 | Moderate                  |
| Interpretability        | High                | Medium                   | Lower                     |
| Best for                | Small datasets      | Medium datasets          | Larger datasets           |

## Usage

```bash
# Independent correlation optimization
python analyzer/main.py --optimize-individual

# Grid sweep with cross-validation
python analyzer/main.py --optimize-combo

# Numerical optimization with cross-validation
python analyzer/main.py --optimize-stepwise

# Live analysis using specific optimization results
python analyzer/main.py                          # uses optimization_results_individual.json (default)
python analyzer/main.py --use-individual         # explicit: uses optimization_results_individual.json
python analyzer/main.py --use-combo              # uses optimization_results_combo.json
python analyzer/main.py --use-stepwise           # uses optimization_results_stepwise.json
python analyzer/main.py --no-opt                 # ignore all optimized weights
```
