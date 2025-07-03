# Technical analysis

A simple Python script that uses a VERY INSECURE, UNOFFICIAL API FROM A SWEDISH BANK THAT NO ONE SHOULD USE to perform technical analysis on stocks in my watchlist.

# Setup

Before using this program, you’ll need to configure a few things related to this UNSECURE, UNOFFICIAL API. I won’t spell it out, but if you read the code, you’ll get an idea of what to search for to find it.

## Instructions
Run analyzer
```bash
python3 analyzer/main.py 
```
Run analyzer and store data
```bash
python3 analyzer/main.py --get_hist True
```
Run analyzer and use stored data
```bash
python3 analyzer/main.py --use_hist True
```
Run metric optimizer
```bash
python3 optimize_metric_generation/optimize_metric_generation.py 
```

# Secret Sauce

The metrics.py file is missing on purpose—it’s where all the magic happens. It contains my secret formula for what makes a stock “good” (or at least what I think makes it good).

# Disclaimer

I have no affiliation with this Swedish bank, nor do I have any formal education in economics. This script is just an experiment, loosely based on things I’ve read from smarter people on the internet. Use at your own risk.