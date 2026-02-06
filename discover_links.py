import os
import requests
from dotenv import load_dotenv

load_dotenv()
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
BASE_URL = "https://www.binance.com/en-IN/binance-api"

OUTPUT_DIR = "data/links"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "binance_api_links.txt")

def discover_links():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    url = "https://api.firecrawl.dev/v1/map"
    headers = {
        "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "url": BASE_URL,
        "includeSubdomains": False,
        "limit": 200
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    links = response.json().get("links", [])

    # Filter only API-related pages
    api_links = sorted({
        link for link in links
        if "binance.com" in link and "api" in link.lower()
    })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for link in api_links:
            f.write(link + "\n")

    print(f"Saved {len(api_links)} links to {OUTPUT_FILE}")

if __name__ == "__main__":
    discover_links()