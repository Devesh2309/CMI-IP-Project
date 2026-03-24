import os
import json
import time
import random
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

from src.config import ISSUES_DIR, DOCS_PROVIDER, SO_TAG, SO_PAGES


def scrape_stackoverflow(
    tag: str = SO_TAG,
    pages: int = SO_PAGES,
    issues_dir: str = ISSUES_DIR,
) -> str:
    os.makedirs(issues_dir, exist_ok=True)
    output_file = os.path.join(issues_dir, f"{DOCS_PROVIDER}_issues.json")

    base_url = "https://stackoverflow.com"
    start_url = f"{base_url}/questions/tagged/{tag}?tab=newest&pagesize=50"

    all_questions = []
    current_url = start_url
    page_count = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,   # ← run headed so SO doesn't fingerprint as bot
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
            locale="en-US",
        )

        # Mask webdriver flag
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

        page = context.new_page()

        while current_url and page_count < pages:
            print(f"\nScraping page {page_count + 1}: {current_url}")

            try:
                page.goto(current_url, wait_until="domcontentloaded", timeout=60000)

                # Random human-like delay between 3–6 seconds
                time.sleep(random.uniform(3, 6))

                # Wait for posts with a generous timeout
                page.wait_for_selector("div.s-post-summary", timeout=30000)

            except PlaywrightTimeoutError:
                print(f"  Timeout on page {page_count + 1} — SO may be blocking. Saving what we have.")
                break
            except Exception as e:
                print(f"  Unexpected error on page {page_count + 1}: {e}")
                break

            posts = page.query_selector_all("div.s-post-summary")
            print(f"  Found {len(posts)} posts")

            for post in posts:
                link_el = post.query_selector("h3 a")
                if not link_el:
                    continue

                title = link_el.inner_text().strip()
                href = link_el.get_attribute("href")
                url = base_url + href if href else ""

                all_questions.append({
                    "title": title,
                    "url": url,
                    "page": page_count + 1,
                })

            page_count += 1

            if page_count < pages:
                next_btn = page.query_selector('a[rel="next"]')
                if next_btn:
                    next_href = next_btn.get_attribute("href")
                    current_url = base_url + next_href
                    # Longer random delay between pages to avoid rate limiting
                    delay = random.uniform(5, 10)
                    print(f"  Waiting {delay:.1f}s before next page...")
                    time.sleep(delay)
                else:
                    print("  No next page found, stopping.")
                    break
            else:
                break

        browser.close()

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=4, ensure_ascii=False)

    print(f"\nSaved {len(all_questions)} questions to {output_file}")
    return output_file


if __name__ == "__main__":
    scrape_stackoverflow()