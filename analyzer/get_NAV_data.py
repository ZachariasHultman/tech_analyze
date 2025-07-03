import requests
import pandas as pd
from io import StringIO

def get_nav_data(ticker_name):
    # URL for the request
    url = "http://ibindex.se/ibi//company/downloadPriceData.req"

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
        "Cookie": "ibi-tracking=a58bf6f7-fbe9-4c72-890a-4bb112fd1dca",
        "Host": "ibindex.se",
        "Origin": "http://ibindex.se",
        "Referer": "http://ibindex.se/ibi/",
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
        print(f"Failed to download file. Status code: {response.status_code}")
        print(f"Response content: {response.text}")