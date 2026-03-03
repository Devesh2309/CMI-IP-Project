import os
import requests

from src.config import FIRECRAWL_API_KEY, BASE_URL, LINKS_DIR


def discover_links(output_filename: str = "api_links.txt", limit: int = 200) -> str:
    """Discover documentation links using Firecrawl and save to file."""

    if not FIRECRAWL_API_KEY:
        raise ValueError("FIRECRAWL_API_KEY not set in environment.")

    os.makedirs(LINKS_DIR, exist_ok=True)
    output_path = os.path.join(LINKS_DIR, output_filename)

    response = requests.post(
        "https://api.firecrawl.dev/v1/map",
        headers={
            "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "url": BASE_URL,
            "includeSubdomains": True,
            "limit": limit,
        },
        timeout=60,
    )

    response.raise_for_status()
    links = response.json().get("links", [])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(links))

    print(f"Saved {len(links)} links to {output_path}")
    return output_path


if __name__ == "__main__":
    discover_links()