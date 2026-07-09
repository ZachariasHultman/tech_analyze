import requests
import pandas as pd
from io import StringIO

def get_nav_data(ticker_name):
    # URL for the request
    url = "https://ibindex.se/ibi//company/downloadPriceData.req"

    # Request payload
    payload = {
        "product": ticker_name,
        "currency": "SEK"
    }

    # Full headers
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "sv-SE,sv;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive",
        "Content-Type": "application/json;charset=UTF-8",
        "Host": "ibindex.se",
        "Origin": "https://ibindex.se",
        "Referer": "https://ibindex.se/ibi/",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
    }

    # Send the POST request
    response = requests.post(url, json=payload, headers=headers)

    # Check the response
    if response.status_code == 200:
        csv_data = StringIO(response.text)  # Convert response text to a file-like object
        df = pd.read_csv(csv_data)
        return df
    else:
        # NAV data unavailable for this ticker (expected for non-Swedish investment companies)
        print(f"[WARN] NAV data request failed for {ticker_name}: status {response.status_code}")
        return None