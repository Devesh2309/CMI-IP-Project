import argparse
from src.scraping.discover_links import discover_links
from src.scraping.render_pages import render_pages
from src.scraping.html_to_markdown import convert_all
from src.graph.build_graph import build
from src.graph.query_issue import query_issue
from src.visualize_graph import visualize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--query", type=str)
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()

    if args.discover:
        discover_links()
    if args.render:
        render_pages()
    if args.markdown:
        convert_all()
    if args.build:
        build()
    if args.query:
        query_issue(args.query)
    if args.visualize:
        visualize(show_chunks=False)

if __name__ == "__main__":
    main()