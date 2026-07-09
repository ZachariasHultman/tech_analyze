import numpy as np
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
    """Write today's snapshot to csv_path.

    Overwrites (not appends) if the file already exists — re-running --save
    the same day replaces the file instead of accumulating duplicate rows.
    """
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
        df.to_csv(csv_path, mode="w", index=False, header=True)
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
    df.to_csv(csv_path, mode="w", index=False, header=True)


def calculate_slope(data, ticker=None):

    x = np.arange(len(data))
    y = np.array(data)

    slope_deg1 = np.polyfit(x, y, 1)[0]

    return slope_deg1
