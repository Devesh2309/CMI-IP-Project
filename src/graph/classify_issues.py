"""
classify_issues.py

Batch-classifies every issue in data/issues/{DOCS_PROVIDER}_issues.json
by running each title through query_issue and packaging the results into
the schema expected by graph_metrics.compute_rag_metrics().

Output
------
Two files written to data/issues/:

  1. {DOCS_PROVIDER}_classified.json   — human-readable, full detail
  2. {DOCS_PROVIDER}_classified.pkl    — pickle for graph_metrics.run()

Each record in the results list has the following keys (matching graph_metrics):

  issue_title      : str          original SO question title
  issue_url        : str          original SO question URL
  result           : str          "Classified" | "Uncertain"
  confidence_score : float        similarity of top-1 match  (0–1)
  node_id          : str          chunk_id of top-1 match
  hierarchy_path   : str          "file root → H1 → H2 → …"
  top_candidates   : list[dict]   [{chunk_id, score, hierarchy_path}, …]
  evidence         : list[str]    hierarchy_path strings of all top_k matches

Run
---
  python main.py --classify

  or directly:
  python src/graph/classify_issues.py
"""

import os
import json
import pickle
from tqdm import tqdm

from src.config import (
    ISSUES_DIR,
    DOCS_PROVIDER,
    TOP_K,
    GRAPH_DIR,
    CHROMA_DIR,
)
from src.graph.query_issue import query_issue


# ── Tuneable threshold ────────────────────────────────────────────────────────
# If top-1 similarity is below this value the issue is marked "Uncertain".
# Cosine similarity in [0, 1] — 0.30 is a reasonable starting point;
# raise it to be stricter, lower it to classify more aggressively.
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.30))


def classify_issues(
    issues_dir: str = ISSUES_DIR,
    docs_provider: str = DOCS_PROVIDER,
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
) -> str:
    """
    Read the issues JSON, classify each issue, write classified JSON + pkl.
    Returns the path to the JSON output file.
    """

    issues_path = os.path.join(issues_dir, f"{docs_provider}_issues.json")

    if not os.path.exists(issues_path):
        raise FileNotFoundError(
            f"Issues file not found: {issues_path}\n"
            f"Run --scrape first to generate it."
        )

    with open(issues_path, "r", encoding="utf-8") as f:
        issues = json.load(f)

    if not issues:
        raise ValueError("Issues file is empty — nothing to classify.")

    print(f"\nClassifying {len(issues)} issues from: {issues_path}")
    print(f"  top_k     = {top_k}")
    print(f"  threshold = {threshold}")
    print("=" * 60)

    classified = []

    for issue in tqdm(issues, desc="Classifying"):
        title = issue.get("title", "").strip()
        url   = issue.get("url",   "")

        if not title:
            continue

        # ── Run vector search + graph walk ────────────────────────────────
        try:
            matches = query_issue(issue_text=title, top_k=top_k)
        except Exception as e:
            # Don't let one bad issue crash the whole run
            print(f"\n  [WARN] query_issue failed for: {title!r} — {e}")
            classified.append(_uncertain_record(title, url))
            continue

        if not matches:
            classified.append(_uncertain_record(title, url))
            continue

        top = matches[0]
        top_similarity = top["similarity"]

        # ── Decide: Classified vs Uncertain ───────────────────────────────
        if top_similarity >= threshold:
            result = "Classified"
        else:
            result = "Uncertain"

        # ── Build top_candidates list (for Metric 8 gap calculation) ──────
        top_candidates = [
            {
                "chunk_id":       m["chunk_id"],
                "score":          m["similarity"],
                "hierarchy_path": m["hierarchy_path"],
            }
            for m in matches
        ]

        # ── evidence = all matched hierarchy paths (for Metric 9) ─────────
        evidence = [m["hierarchy_path"] for m in matches if m["hierarchy_path"]]

        record = {
            # ── issue identity ────────────────────────────────────────────
            "issue_title":      title,
            "issue_url":        url,

            # ── classification result (Metric 6) ─────────────────────────
            "result":           result,

            # ── top-1 detail (Metrics 7, 10) ─────────────────────────────
            "confidence_score": top_similarity,
            "node_id":          top["chunk_id"],
            "hierarchy_path":   top["hierarchy_path"],
            "hierarchy":        top["hierarchy"],

            # ── full ranked list (Metric 8) ───────────────────────────────
            "top_candidates":   top_candidates,

            # ── evidence list (Metric 9) ──────────────────────────────────
            "evidence":         evidence,
        }

        classified.append(record)

    # Summary 
    total       = len(classified)
    n_classified = sum(1 for r in classified if r["result"] == "Classified")
    n_uncertain  = sum(1 for r in classified if r["result"] == "Uncertain")

    print(f"\nDone.")
    print(f"  Total     : {total}")
    print(f"  Classified: {n_classified}  ({n_classified/total:.1%})")
    print(f"  Uncertain : {n_uncertain}  ({n_uncertain/total:.1%})")

    # Write JSON (human-readable)
    json_out = os.path.join(issues_dir, f"{docs_provider}_classified.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(classified, f, indent=4, ensure_ascii=False)
    print(f"\n  JSON  → {json_out}")

    # Write pkl (for graph_metrics.run())
    pkl_out = os.path.join(issues_dir, f"{docs_provider}_classified.pkl")
    with open(pkl_out, "wb") as f:
        pickle.dump(classified, f)
    print(f"  PKL   → {pkl_out}")

    return json_out


# Helpers
def _uncertain_record(title: str, url: str) -> dict:
    """Return a minimal Uncertain record when query_issue returns nothing."""
    return {
        "issue_title":      title,
        "issue_url":        url,
        "result":           "Uncertain",
        "confidence_score": 0.0,
        "node_id":          None,
        "hierarchy_path":   "",
        "hierarchy":        [],
        "top_candidates":   [],
        "evidence":         [],
    }


if __name__ == "__main__":
    classify_issues()