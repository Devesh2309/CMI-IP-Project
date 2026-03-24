import argparse
from src.scraping.discover_links import discover_links
from src.scraping.render_pages import render_pages
from src.scraping.html_to_markdown import convert_all
from src.graph.build_graph import build
from src.graph.query_issue import query_issue
from src.graph.classify_issues import classify_issues
from src.visualize_graph import visualize


def main():
    parser = argparse.ArgumentParser(
        description="API Feature Knowledge Graph — pipeline runner"
    )

    # ── Pipeline steps ────────────────────────────────────────────────────────
    parser.add_argument("--discover",  action="store_true", help="Discover doc links via Firecrawl")
    parser.add_argument("--render",    action="store_true", help="Render pages to HTML via Playwright")
    parser.add_argument("--markdown",  action="store_true", help="Convert HTML → Markdown")
    parser.add_argument("--build",     action="store_true", help="Build feature graph + Chroma DB")
    parser.add_argument("--scrape",    action="store_true", help="Scrape Stack Overflow issues")
    parser.add_argument("--classify",  action="store_true", help="Classify all issues in issues JSON")
    parser.add_argument("--metrics",   action="store_true", help="Print graph + RAG metrics")
    parser.add_argument("--visualize", action="store_true", help="Generate PyVis graph HTML")

    # ── Per-command options ───────────────────────────────────────────────────
    parser.add_argument("--query",     type=str,            help="Classify a single issue text")
    parser.add_argument("--so-tag",    type=str, default=None, help="Override SO tag for --scrape")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Similarity threshold for --classify (default: 0.30)")

    args = parser.parse_args()

    # ── Run steps ─────────────────────────────────────────────────────────────
    if args.discover:
        discover_links()

    if args.render:
        render_pages()

    if args.markdown:
        convert_all()

    if args.build:
        build()

    if args.scrape:
        from src.scraping.scrape_stackoverflow import scrape_stackoverflow
        from src.config import SO_TAG, SO_PAGES
        tag = args.so_tag if args.so_tag else SO_TAG
        scrape_stackoverflow(tag=tag, pages=SO_PAGES)

    if args.classify:
        kwargs = {}
        if args.threshold is not None:
            kwargs["threshold"] = args.threshold
        classify_issues(**kwargs)

    if args.metrics:
        import os
        from src.graph.graph_metrics import run as run_metrics
        from src.config import ISSUES_DIR, DOCS_PROVIDER
        pkl_path = os.path.join(ISSUES_DIR, f"{DOCS_PROVIDER}_classified.pkl")
        results_path = pkl_path if os.path.exists(pkl_path) else None
        run_metrics(results_path=results_path)

    if args.query:
        query_issue(args.query)

    if args.visualize:
        visualize(show_chunks=False)


if __name__ == "__main__":
    main()