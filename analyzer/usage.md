# Usage

python main.py	Live analysis. Automatically uses optimized weights from optimization_results.json if it exists. Shows reliability column + columns ordered by weight.

python main.py --save	Same as above + saves data snapshots to data/ for future optimization.

python main.py --correlate	Shows per-metric Spearman correlation with stock returns (uses saved historical data from data/).

python main.py --optimize	Runs correlation + optimizes weights + saves to optimization_results.json and company_reliability.csv. Next live run will automatically pick up these weights.

python main.py --no-opt	Live analysis but ignores optimized weights, uses the hardcoded defaults from metrics.py.


# Typical workflow
```bash
python main.py --save — run periodically to collect data snapshots
python main.py --optimize-individual — once you have enough snapshots, optimize weights or
python main.py --optimize-combo — once you have enough snapshots, optimize weights or
python main.py --optimize-stepwise
python analyzer/main.py --use-individual         # explicit: uses optimization_results_individual.json
python analyzer/main.py --use-combo              # uses optimization_results_combo.json
python analyzer/main.py --use-stepwise           # uses optimization_results_stepwise.json
python analyzer/main.py --no-opt                 # ignore all optimized weights and thresholds
````
