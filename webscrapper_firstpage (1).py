from playwright.sync_api import sync_playwright
import json
import time

def scrape_stackoverflow():
    url = "https://stackoverflow.com/questions/tagged/notion-api?tab=newest&pagesize=50"
    #"https://stackoverflow.com/questions/tagged/discord?tab=newest&page=9&pagesize=50"
    #"https://stackoverflow.com/questions/tagged/slack?tab=newest&page=5&pagesize=50" 
    #"https://stackoverflow.com/questions/tagged/amazon-s3?tab=newest&page=19&pagesize=50"
    #"https://stackoverflow.com/questions/tagged/amazon-s3?tab=newest&pagesize=50"
    #"https://stackoverflow.com/questions/tagged/twilio?tab=newest&page=20&pagesize=50"
    #"https://stackoverflow.com/questions/tagged/twilio?tab=newest&pagesize=50"
    
    questions = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Set True after testing
        page = browser.new_page()
        page.goto(url, timeout=120000)

        page.wait_for_selector("div.s-post-summary")

        posts = page.query_selector_all("div.s-post-summary")

        for post in posts:
            title = post.query_selector("h3 a").inner_text()
            link = post.query_selector("h3 a").get_attribute("href")
            full_url = "https://stackoverflow.com" + link

            questions.append({
                "title": title,
                "url": full_url
            })

        browser.close()

    with open("slack_questions10.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=4)

    print("✅ Done. Saved file.")

if __name__ == "__main__":
    scrape_stackoverflow()