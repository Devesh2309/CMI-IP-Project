import os
import re
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

from src.config import HTML_DIR, LINKS_FILE


def safe_filename(url: str) -> str:
    path = urlparse(url).path.strip("/") or "index"
    path = re.sub(r'[<>:"/\\|?*%]', "_", path)
    return path + ".html"


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


def is_junk_url(url: str) -> bool:
    junk_patterns = []
    return any(p in url for p in junk_patterns)


def render_pages(
    html_dir: str = HTML_DIR,
    headless: bool = True,
) -> None:
    """Render documentation pages using Playwright and save cleaned HTML."""

    links_path = LINKS_FILE

    if not os.path.exists(links_path):
        raise FileNotFoundError(f"Links file not found: {links_path}")

    os.makedirs(html_dir, exist_ok=True)

    with open(links_path, "r", encoding="utf-8") as f:
        urls = [u.strip() for u in f if u.strip()]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
        )

        for url in urls:
            print(f"\n Rendering {url}")

            # Skip junk URLs
            if is_junk_url(url):
                print("⏭ Skipping junk URL")
                continue

            try:
                # fresh page per URL
                page = context.new_page()
                page.goto(url, wait_until="networkidle", timeout=60000)
                visible_text = page.locator("body").inner_text()

                #  Retry if JS not loaded properly
                if "Enable JavaScript" in visible_text or len(visible_text.strip()) < 200:
                    print("⚠ JS issue detected, retrying...")
                    page.goto(url, wait_until="networkidle", timeout=60000)
                    visible_text = page.locator("body").inner_text()

                #  Skip blocked pages
                if "Enable JavaScript" in visible_text:
                    print(" Blocked page, skipping")
                    page.close()
                    continue

                content_html = extract_main_content(page)

                filename = safe_filename(url)
                output_path = os.path.join(html_dir, filename)

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content_html)

                print(f" Saved: {filename}")

                page.close()

            except Exception as e:
                print(f" Error on {url}: {e}")
                continue
        browser.close()

    print(f"\n HTML saved to {html_dir}")


if __name__ == "__main__":
    render_pages()