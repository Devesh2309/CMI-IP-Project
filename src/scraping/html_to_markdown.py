import os
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from src.config import HTML_DIR, MARKDOWN_DIR


def convert_all(html_dir: str = HTML_DIR, markdown_dir: str = MARKDOWN_DIR) -> None:
    """Convert all HTML files in a directory to cleaned Markdown."""

    if not os.path.exists(html_dir):
        raise FileNotFoundError(f"HTML directory not found: {html_dir}")

    os.makedirs(markdown_dir, exist_ok=True)

    for filename in os.listdir(html_dir):
        if not filename.endswith(".html"):
            continue

        html_path = os.path.join(html_dir, filename)

        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()

        soup = BeautifulSoup(html, "lxml")

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

        output_filename = filename.replace(".html", ".md")
        output_path = os.path.join(markdown_dir, output_filename)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        print(f"Converted {filename} → {output_filename}")


if __name__ == "__main__":
    convert_all()