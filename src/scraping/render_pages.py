import os
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

from src.config import LINKS_DIR, HTML_DIR


def safe_filename(url: str) -> str:
    path = urlparse(url).path.strip("/")
    return (path.replace("/", "_") or "index") + ".html"


def extract_main_content(page) -> str:
    selectors = [
        "article",
        "main",
        "[role=main]",
        "#content",
        ".content",
        ".documentation",
    ]

    for selector in selectors:
        locator = page.locator(selector)
        if locator.count() > 0:
            html = locator.first.inner_html()
            if len(html.strip()) > 500:
                print(f"Using selector: {selector}")
                return html

    print("Falling back to <body>")
    return page.locator("body").inner_html()


def render_pages(
    links_filename: str = "api_links.txt",
    html_dir: str = HTML_DIR,
    headless: bool = True,
) -> None:
    """Render documentation pages using Playwright and save cleaned HTML."""

    links_path = os.path.join(LINKS_DIR, links_filename)

    if not os.path.exists(links_path):
        raise FileNotFoundError(f"Links file not found: {links_path}")

    os.makedirs(html_dir, exist_ok=True)

    with open(links_path, "r", encoding="utf-8") as f:
        urls = [u.strip() for u in f if u.strip()]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()

        for url in urls:
            print(f"Rendering {url}")
            page.goto(url, wait_until="domcontentloaded", timeout=60000)

            content_html = extract_main_content(page)
            filename = safe_filename(url)
            output_path = os.path.join(html_dir, filename)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content_html)

        browser.close()

    print(f"HTML saved to {html_dir}")


if __name__ == "__main__":
    render_pages()