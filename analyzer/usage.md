# Usage

python main.py	Live analysis. Automatically uses optimized weights from optimization_results.json if it exists. Shows reliability column + columns ordered by weight.

python main.py --save	Same as above + saves data snapshots to data/ for future optimization.

python main.py --correlate	Shows per-metric Spearman correlation with stock returns (uses saved historical data from data/).

python main.py --optimize	Runs correlation + optimizes weights + saves to optimization_results.json and company_reliability.csv. Next live run will automatically pick up these weights.

python main.py --no-opt	Live analysis but ignores optimized weights, uses the hardcoded defaults from metrics.py.


# Typical workflow

python main.py --save — run periodically to collect data snapshots
python main.py --optimize — once you have enough snapshots, optimize weights
python main.py — from now on, live analysis uses the optimized weights automatically