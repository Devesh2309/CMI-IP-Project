from playwright.sync_api import sync_playwright
import json
import os
import time
from src.config import ISSUES_DIR, DOCS_PROVIDER

def scrape_stackoverflow():
    # Uses DOCS_PROVIDER as the tag (e.g., 'discord' or 'notion-api')
    tag = DOCS_PROVIDER
    base_url = f"https://stackoverflow.com/questions/tagged/{tag}?tab=newest&pagesize=50"
    output_path = os.path.join(ISSUES_DIR, f"{tag}_issues.json")
    questions = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        current_url = base_url
        page_count = 1

        while current_url:
            print(f"Scraping page {page_count}...")
            page.goto(current_url, timeout=120000)
            page.wait_for_selector("div.s-post-summary")

            posts = page.query_selector_all("div.s-post-summary")
            for post in posts:
                title_elem = post.query_selector("h3 a")
                questions.append({
                    "title": title_elem.inner_text(),
                    "url": "https://stackoverflow.com" + title_elem.get_attribute("href")
                })

            # Check for "Next" button
            next_button = page.query_selector('a[rel="next"]')
            if next_button:
                current_url = "https://stackoverflow.com" + next_button.get_attribute("href")
                page_count += 1
                time.sleep(1) # Polite delay
            else:
                current_url = None

        browser.close()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=4)

    print(f"✅ Done. Saved {len(questions)} issues to {output_path}")

if __name__ == "__main__":
    scrape_stackoverflow()