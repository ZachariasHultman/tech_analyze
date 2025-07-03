import numpy as np
import matplotlib.pyplot as plt
from avanza.models import *
import pandas as pd
from pathlib import Path
from datetime import date


# helper.py
from pathlib import Path
import json


from pathlib import Path
import json
import pandas as pd
from datetime import date, datetime
import numpy as np


def save_snapshot(data, csv_path, asof):
    csv_path = Path(csv_path)

    # If data is a DataFrame, convert any datetime columns to ISO strings and write directly.
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or (
                df[col].dtype == object
                and df[col]
                .dropna()
                .apply(lambda x: isinstance(x, (date, datetime)))
                .all()
            ):
                df[col] = df[col].astype(str)
        df.insert(0, "asof", asof)
        header = not csv_path.exists()
        df.to_csv(csv_path, mode="a", index=False, header=header)
        return

    # Otherwise, data is a dict. Build a one‐row dict of JSON‐encoded strings.
    row = {}
    for k, v in data.items():
        # 1) If v is a DataFrame → convert to list of records, stringify dates
        if isinstance(v, pd.DataFrame):
            records = v.to_dict("records")
            for rec in records:
                for entry_k, entry_v in rec.items():
                    if isinstance(entry_v, (date, datetime)):
                        rec[entry_k] = entry_v.isoformat()
            row[k] = json.dumps(records)
            continue

        # 2) If v is a numpy array → convert to Python list
        if isinstance(v, np.ndarray):
            v = v.tolist()

        # 3) If v is a list of dicts → stringify any dates inside each dict
        if isinstance(v, list) and v and isinstance(v[0], dict):
            normalized = []
            for entry in v:
                new_entry = {}
                for entry_k, entry_v in entry.items():
                    if isinstance(entry_v, (date, datetime)):
                        new_entry[entry_k] = entry_v.isoformat()
                    else:
                        new_entry[entry_k] = entry_v
                normalized.append(new_entry)
            row[k] = json.dumps(normalized)
            continue

        # 4) Otherwise (scalar or list of scalars) → JSON‐encode directly
        row[k] = json.dumps(v)

    df = pd.DataFrame([row])
    df.insert(0, "asof", asof)
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", index=False, header=header)


def z_score(data):
    mean_val = np.mean(data)
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - mean_val) / (max_val - min_val) for x in data]
    return normalized_data


def plot_data_with_fit(data):
    x = np.arange(len(data))
    y = np.array(data)
    slope_deg1 = np.polyfit(x, y, 1)
    slope_deg2 = np.polyfit(x, y, 2)
    slope_deg3 = np.polyfit(x, y, 3)
    # Generate polynomial functions
    poly_deg1 = np.poly1d(slope_deg1)
    poly_deg2 = np.poly1d(slope_deg2)
    poly_deg3 = np.poly1d(slope_deg3)

    # Create a smooth range for x to plot the polynomials
    x_smooth = np.linspace(min(x), max(x), 500)

    # Calculate polynomial values
    y_deg1 = poly_deg1(x_smooth)
    y_deg2 = poly_deg2(x_smooth)
    y_deg3 = poly_deg3(x_smooth)

    # Plot original data
    plt.scatter(x, y, label="Original Data", color="black", s=10)

    # Plot polynomial fits
    plt.plot(x_smooth, y_deg1, label="Degree 1 Fit", color="red", linestyle="--")
    plt.plot(x_smooth, y_deg2, label="Degree 2 Fit", color="blue", linestyle="-.")
    plt.plot(x_smooth, y_deg3, label="Degree 3 Fit", color="green")

    # Add labels and legend
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Polynomial Fits")
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()


def calculate_slope(data, ticker=None):

    x = np.arange(len(data))
    y = np.array(data)

    slope_deg1 = np.polyfit(x, y, 1)[0]

    return slope_deg1
