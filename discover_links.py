import os
import requests
from dotenv import load_dotenv

load_dotenv()
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
BASE_URL = "https://docs.stripe.com/get-started"

OUTPUT_DIR = "data/links"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "stripe_api_links.txt")

def discover_links():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    url = "https://api.firecrawl.dev/v1/map"
    headers = {
        "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "url": BASE_URL,
        "includeSubdomains": True,
        "limit": 200
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    links = response.json().get("links", [])

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for link in links:
            f.write(link + "\n")

    print(f"Saved {len(links)} links to {OUTPUT_FILE}")

if __name__ == "__main__":
    discover_links()