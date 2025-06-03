import numpy as np
import matplotlib.pyplot as plt
from avanza.models import *
import pandas as pd
from pathlib import Path
from datetime import date


def save_snapshot(data: dict, csv_path, asof):
    csv_path = Path(csv_path)  # convert string → Path
    df = pd.DataFrame([data])
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

    slope_deg1 = np.polyfit(x, y, 1)
    slope_deg2 = np.polyfit(x, y, 2)
    slope_deg3 = np.polyfit(x, y, 3)

    # only use the highest degree coeficcient to get the trend
    slopes = [slope_deg1[0], slope_deg2[0], slope_deg3[0]]

    if len(data) <= 20:
        return slope_deg1[0]

    # Count positive and negative slopes
    positive_count = sum(1 for slope in slopes if slope >= 0)

    # Check if 2 or more slopes are positive or negative
    if positive_count >= 2:
        # Return the highest-degree slope that is positive
        for slope in reversed(slopes):  # Iterate from degree 3 to 1
            if slope >= 0:
                return slope
    else:
        # Return the highest-degree slope that is negative
        for slope in reversed(slopes):  # Iterate from degree 3 to 1
            if slope < 0:
                return slope
