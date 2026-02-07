import os
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

LINKS_FILE = "data/links/stripe_api_links.txt"
HTML_DIR = "data/rendered_html"

os.makedirs(HTML_DIR, exist_ok=True)

def safe_filename(url):
    path = urlparse(url).path.strip("/")
    return (path.replace("/", "_") or "index") + ".html"

with open(LINKS_FILE, "r", encoding="utf-8") as f:
    urls = [u.strip() for u in f if u.strip()]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    for url in urls:
        print(f"Rendering {url}")
        page.goto(url, wait_until="domcontentloaded", timeout=60000)

        # Stripe content is inside <article>
        article_html = page.locator("article").inner_html()

        filename = safe_filename(url)
        with open(os.path.join(HTML_DIR, filename), "w", encoding="utf-8") as f:
            f.write(article_html)

    browser.close()

print("Done! HTML saved to", HTML_DIR)
