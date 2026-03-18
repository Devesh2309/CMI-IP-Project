import os
from bs4 import BeautifulSoup
from markdownify import markdownify as md

HTML_DIR = "data/rendered_html"
MD_DIR = "data/markdown"

os.makedirs(MD_DIR, exist_ok=True)

for file in os.listdir(HTML_DIR):
    if not file.endswith(".html"):
        continue

    with open(os.path.join(HTML_DIR, file), "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "lxml")

    # Remove UI-only junk
    for tag in soup.select(
        "button, svg, nav, footer, aside, "
        "[role=toolbar], [aria-hidden=true]"
    ):
        tag.decompose()

    markdown = md(
        str(soup),
        heading_style="ATX",
        code_language_callback=lambda _: ""
    )

    out_file = file.replace(".html", ".md")
    with open(os.path.join(MD_DIR, out_file), "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"Converted {file} → {out_file}")