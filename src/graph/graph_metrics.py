"""
graph_metrics.py

Computes all 10 metrics that match the reference notebook exactly.

GRAPH METRICS  (describe the quality of the feature hierarchy)
  Metric 1  — Root Ratio
  Metric 2  — Subtree Size Distribution
  Metric 3  — Max / Avg Depth
  Metric 4  — Support Count Distribution
  Metric 5  — Heading Level Distribution

RAG METRICS  (describe how well issue classification performed)
  Metric 6  — Uncertain Rate
  Metric 7  — Average Confidence Score
  Metric 8  — Top-1 vs Top-2 Confidence Gap
  Metric 9  — Evidence Count Per Classification
  Metric 10 — Node Coverage by Issues

Run standalone:
    python graph_metrics.py

Or via main.py:
    python main.py --metrics
"""

import os
import pickle
import json
from collections import Counter

import pandas as pd
import networkx as nx

from src.config import GRAPH_DIR, ISSUES_DIR, DOCS_PROVIDER


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_graph(graph_filename: str = "feature_graph.pkl") -> nx.DiGraph:
    graph_path = os.path.join(GRAPH_DIR, graph_filename)
    if not os.path.exists(graph_path):
        raise FileNotFoundError(
            f"Graph not found at {graph_path}\n"
            f"Run --build first."
        )
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    print(f"Loaded graph:  {graph_path}")
    print(f"  Total nodes : {G.number_of_nodes()}")
    print(f"  Total edges : {G.number_of_edges()}")
    return G


def get_feature_subgraph(G: nx.DiGraph) -> nx.DiGraph:
    """Return a copy of G containing only feature nodes (no UUID chunk nodes)."""
    feature_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "feature"]
    return G.subgraph(feature_nodes).copy()


def build_nodes_df(H: nx.DiGraph) -> pd.DataFrame:
    """
    Build a DataFrame of feature nodes.

    Metric 4 fix — support_count:
      Node IDs are 'filename::heading', so the same heading in two files
      produces two separate nodes.  We compute support_count as the number
      of distinct source files in which a given heading label appears.
      This matches the notebook's definition (occurrences across files).
    """
    # Count how many files each heading label appears in
    label_file_sets: dict[str, set] = {}
    for node, attr in H.nodes(data=True):
        label = attr.get("label", node)
        src   = attr.get("source_file", "")
        label_file_sets.setdefault(label, set()).add(src)

    label_support = {lbl: len(files) for lbl, files in label_file_sets.items()}

    rows = []
    for node, attr in H.nodes(data=True):
        label = attr.get("label", node)
        rows.append({
            "node_id":       node,
            "name":          label,
            "level":         attr.get("level", None),
            "source_file":   attr.get("source_file", ""),
            "support_count": label_support.get(label, 1),
        })

    return pd.DataFrame(rows)


def subtree_stats(G: nx.DiGraph, root: str) -> tuple[int, int]:
    """Return (subtree_size, max_depth) for a root node via iterative DFS."""
    visited   = set()
    stack     = [(root, 0)]
    max_depth = 0
    while stack:
        node, depth = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        max_depth = max(max_depth, depth)
        for child in G.successors(node):
            stack.append((child, depth + 1))
    return len(visited), max_depth


def _div(a, b, fallback=0):
    return a / b if b else fallback


def _sep(title: str = ""):
    line = "=" * 55
    if title:
        print(f"\n{line}")
        print(f"  {title}")
        print(line)
    else:
        print(line)


# ──────────────────────────────────────────────────────────────────────────────
# GRAPH METRICS  (Metrics 1 – 5)
# ──────────────────────────────────────────────────────────────────────────────

def compute_graph_metrics(G: nx.DiGraph):
    """
    Compute and print Metrics 1–5 from the feature subgraph.
    Returns a dict of computed values for use in the summary table.
    """

    H = get_feature_subgraph(G)

    total_nodes = H.number_of_nodes()
    roots       = [n for n in H.nodes() if H.in_degree(n) == 0]

    # Pre-compute subtree stats for every root (used by Metrics 2 & 3)
    root_metrics = {r: subtree_stats(H, r) for r in roots}

    nodes_df = build_nodes_df(H)

    # ── Metric 1: Root Ratio ──────────────────────────────────────────────────
    root_ratio = _div(len(roots), total_nodes)

    _sep("METRIC 1 — Root Ratio")
    print(f"  Total nodes  : {total_nodes}")
    print(f"  Root nodes   : {len(roots)}")
    print(f"  Root ratio   : {root_ratio:.1%}")

    if root_ratio < 0.30:
        m1_status = "GOOD"
        print("  STATUS → GOOD  (< 30% — well-connected hierarchy)")
    elif root_ratio < 0.50:
        m1_status = "OK"
        print("  STATUS → ACCEPTABLE  (30–50%, some fragmentation)")
    else:
        m1_status = "BAD"
        print("  STATUS → BAD  (> 50% — graph is a flat list, not a hierarchy)")

    # ── Metric 2: Subtree Size Distribution ───────────────────────────────────
    subtree_sizes = [root_metrics[r][0] for r in roots]

    if subtree_sizes:
        max_subtree  = max(subtree_sizes)
        min_subtree  = min(subtree_sizes)
        avg_subtree  = sum(subtree_sizes) / len(subtree_sizes)
        giant_share  = _div(max_subtree, total_nodes)
        largest_root = roots[subtree_sizes.index(max_subtree)]
        largest_name = H.nodes[largest_root].get("label", largest_root)
    else:
        max_subtree = min_subtree = avg_subtree = giant_share = 0
        largest_name = "N/A"

    _sep("METRIC 2 — Subtree Size Distribution")
    print(f"  Number of roots        : {len(roots)}")
    print(f"  Avg subtree size       : {avg_subtree:.1f} nodes")
    print(f"  Largest subtree        : {max_subtree} nodes  ('{largest_name}')")
    print(f"  Smallest subtree       : {min_subtree} nodes")
    print(f"  Largest share of graph : {giant_share:.1%}")

    if giant_share < 0.40:
        m2_status = "GOOD"
        print("  STATUS → GOOD  (no single root dominates)")
    elif giant_share < 0.60:
        m2_status = "OK"
        print("  STATUS → ACCEPTABLE  (one root is large but not overwhelming)")
    else:
        m2_status = "BAD"
        print("  STATUS → BAD  (> 60% — one bucket dominates, routing will be useless)")

    # Top 5 roots by subtree size
    print("\n  Top 5 roots by subtree size:")
    paired = sorted(zip(subtree_sizes, roots), reverse=True)
    for size, r in paired[:5]:
        name  = H.nodes[r].get("label", r)
        depth = root_metrics[r][1]
        share = _div(size, total_nodes) * 100
        print(f"    {size:>4} nodes ({share:4.1f}%)  depth={depth}  →  {name}")

    # ── Metric 3: Depth ───────────────────────────────────────────────────────
    depths = [root_metrics[r][1] for r in roots]

    if depths:
        max_depth = max(depths)
        avg_depth = sum(depths) / len(depths)
        min_depth = min(depths)
    else:
        max_depth = avg_depth = min_depth = 0

    _sep("METRIC 3 — Max Depth")
    print(f"  Max depth across all roots : {max_depth}")
    print(f"  Avg depth across roots     : {avg_depth:.2f}")
    print(f"  Min depth across roots     : {min_depth}")

    if 3 <= avg_depth <= 5:
        m3_status = "GOOD"
        print("  STATUS → GOOD  (avg depth 3–5)")
    elif avg_depth < 3:
        m3_status = "BAD"
        print("  STATUS → BAD  (< 3 — hierarchy too shallow, features too vague)")
    else:
        m3_status = "BAD"
        print("  STATUS → BAD  (> 5 — hierarchy too deep, features too granular)")

    # Depth histogram
    depth_hist = Counter(depths)
    print("\n  Distribution of root depths:")
    for d in sorted(depth_hist):
        bar = "█" * min(depth_hist[d], 80)   # cap bar at 80 chars
        print(f"    depth {d}  |  {bar}  ({depth_hist[d]} roots)")

    # ── Metric 4: Support Count Distribution ──────────────────────────────────
    _sep("METRIC 4 — Support Count Distribution")

    if "support_count" in nodes_df.columns and not nodes_df.empty:
        total_nodes_df = len(nodes_df)
        single_occ     = (nodes_df["support_count"] == 1).sum()
        single_ratio   = _div(single_occ, total_nodes_df)
        avg_support    = nodes_df["support_count"].mean()
        median_support = nodes_df["support_count"].median()
        max_support    = nodes_df["support_count"].max()
        top_node_name  = nodes_df.loc[nodes_df["support_count"].idxmax(), "name"]

        print(f"  Total nodes              : {total_nodes_df}")
        print(f"  Nodes seen only once     : {single_occ}  ({single_ratio:.1%})")
        print(f"  Avg support count        : {avg_support:.2f}")
        print(f"  Median support count     : {median_support:.1f}")
        print(f"  Max support count        : {max_support}  ('{top_node_name}')")

        if single_ratio < 0.40:
            m4_status = "GOOD"
            print("  STATUS → GOOD  (< 40% single-occurrence nodes)")
        elif single_ratio < 0.60:
            m4_status = "OK"
            print("  STATUS → ACCEPTABLE  (40–60%, moderate noise)")
        else:
            m4_status = "BAD"
            print("  STATUS → BAD  (> 60% — graph is mostly noise headings)")

        # Support count bucket distribution
        print("\n  Support count buckets:")
        buckets = [(1, 1), (2, 3), (4, 9), (10, 49), (50, 9999)]
        for lo, hi in buckets:
            count = ((nodes_df["support_count"] >= lo) & (nodes_df["support_count"] <= hi)).sum()
            label = (f"seen {lo}" if lo == hi
                     else (f"seen {lo}–{hi}" if hi < 9999 else f"seen {lo}+"))
            bar   = "█" * min(count, 40)
            print(f"    {label:<12}  |  {bar}  ({count})")
    else:
        m4_status = "N/A"
        single_ratio = None
        print("  support_count column not available.")

    # ── Metric 5: Heading Level Distribution ──────────────────────────────────
    _sep("METRIC 5 — Heading Level Distribution")

    if "level" in nodes_df.columns and nodes_df["level"].notna().any():
        level_counts     = nodes_df["level"].value_counts().sort_index()
        total_with_level = level_counts.sum()
        peak_level       = int(level_counts.idxmax())

        print(f"  {'Level':<10} {'Count':>8} {'Share':>8}  Bar")
        print(f"  {'-'*10} {'-'*8} {'-'*8}  ---")
        for lvl, cnt in level_counts.items():
            share = _div(cnt, total_with_level) * 100
            bar   = "█" * int(share / 2)   # 50% = 25 blocks
            print(f"  H{int(lvl):<9} {cnt:>8} {share:>7.1f}%  {bar}")

        print()
        if peak_level == 3:
            m5_status = "GOOD"
            print("  STATUS → GOOD  (H3 is the most common — classic pyramid shape)")
        elif peak_level == 2:
            m5_status = "OK"
            print("  STATUS → ACCEPTABLE  (H2 dominates — slightly shallow)")
        elif peak_level == 1:
            m5_status = "BAD"
            print("  STATUS → BAD  (H1 dominates — graph is too flat)")
        else:
            m5_status = "BAD"
            print(f"  STATUS → BAD  (H{peak_level} dominates — too deep / capturing noise)")
    else:
        peak_level = None
        m5_status  = "N/A"
        print("  'level' column not available in nodes_df.")

    # ── Graph Metrics Summary Table ────────────────────────────────────────────
    print("\n")
    _sep("GRAPH METRICS SUMMARY")

    single_ratio_str = f"{single_ratio:.1%}" if single_ratio is not None else "N/A"
    peak_str         = f"H{peak_level}"       if peak_level  is not None else "N/A"

    col_w = [35, 15, 8]
    rows  = [
        ["Metric",                       "Value",              "Status"],
        ["Root ratio",                   f"{root_ratio:.1%}",  m1_status],
        ["Largest subtree share",        f"{giant_share:.1%}", m2_status],
        ["Avg hierarchy depth",          f"{avg_depth:.2f}",   m3_status],
        ["Single-occurrence node ratio", single_ratio_str,     m4_status],
        ["Peak heading level",           peak_str,             m5_status],
    ]
    for row in rows:
        print(f"  {row[0]:<{col_w[0]}} {row[1]:<{col_w[1]}} {row[2]}")

    return {
        "root_ratio":    root_ratio,
        "giant_share":   giant_share,
        "avg_depth":     avg_depth,
        "single_ratio":  single_ratio,
        "peak_level":    peak_level,
    }


# ──────────────────────────────────────────────────────────────────────────────
# RAG METRICS  (Metrics 6 – 10)
# ──────────────────────────────────────────────────────────────────────────────

def compute_rag_metrics(results: list, G: nx.DiGraph):
    """
    Compute and print Metrics 6–10 from the classified results list.

    Each item in `results` must have:
        result           : str   "Classified" | "Uncertain"
        confidence_score : float
        node_id          : str | None
        top_candidates   : list[{score: float, ...}]
        evidence         : list

    Returns a dict of computed values for the summary table.
    """

    # Feature-only subgraph for correct node count in Metric 10
    H           = get_feature_subgraph(G)
    total_nodes = H.number_of_nodes()

    total_issues    = len(results)
    uncertain_count = sum(1 for r in results if r.get("result") == "Uncertain")
    uncertain_rate  = _div(uncertain_count, total_issues)

    # ── Metric 6: Uncertain Rate ───────────────────────────────────────────────
    _sep("METRIC 6 — Uncertain Rate")
    print(f"  Total issues classified : {total_issues}")
    print(f"  Returned 'Uncertain'    : {uncertain_count}")
    print(f"  Uncertain rate          : {uncertain_rate:.1%}")

    if uncertain_rate < 0.20:
        m6_status = "GOOD"
        print("  STATUS → GOOD  (< 20%)")
    elif uncertain_rate < 0.40:
        m6_status = "OK"
        print("  STATUS → ACCEPTABLE  (20–40%, watch threshold)")
    else:
        m6_status = "BAD"
        print("  STATUS → BAD  (> 40%, corpus gap or threshold too strict)")

    accepted = [r for r in results if r.get("result") != "Uncertain"]

    # ── Metric 7: Average Confidence Score ────────────────────────────────────
    if accepted:
        avg_conf = sum(r["confidence_score"] for r in accepted) / len(accepted)
        min_conf = min(r["confidence_score"] for r in accepted)
        max_conf = max(r["confidence_score"] for r in accepted)
    else:
        avg_conf = min_conf = max_conf = 0.0

    _sep("METRIC 7 — Average Confidence Score  (accepted only)")
    print(f"  Accepted classifications : {len(accepted)}")
    print(f"  Avg confidence score     : {avg_conf:.4f}")
    print(f"  Min / Max                : {min_conf:.4f} / {max_conf:.4f}")

    if avg_conf > 0.45:
        m7_status = "GOOD"
        print("  STATUS → GOOD  (> 0.45)")
    elif avg_conf > 0.35:
        m7_status = "OK"
        print("  STATUS → ACCEPTABLE  (0.35–0.45, consider better embeddings)")
    else:
        m7_status = "BAD"
        print("  STATUS → BAD  (< 0.35, weak matches — not grounded)")

    # ── Metric 8: Top-1 vs Top-2 Confidence Gap ───────────────────────────────
    gaps = []
    for r in accepted:
        cands = r.get("top_candidates", [])
        if len(cands) >= 2:
            gaps.append(cands[0]["score"] - cands[1]["score"])

    avg_gap = sum(gaps) / len(gaps) if gaps else 0.0
    min_gap = min(gaps)             if gaps else 0.0
    max_gap = max(gaps)             if gaps else 0.0

    _sep("METRIC 8 — Top-1 vs Top-2 Confidence Gap")
    print(f"  Issues with 2+ candidates : {len(gaps)}")
    print(f"  Avg gap                   : {avg_gap:.4f}")
    print(f"  Min / Max gap             : {min_gap:.4f} / {max_gap:.4f}")

    if avg_gap > 0.08:
        m8_status = "GOOD"
        print("  STATUS → GOOD  (avg gap > 0.08 — decisive classifications)")
    elif avg_gap > 0.03:
        m8_status = "OK"
        print("  STATUS → ACCEPTABLE  (0.03–0.08, some ambiguity)")
    else:
        m8_status = "BAD"
        print("  STATUS → BAD  (< 0.03 — classifier is effectively guessing)")

    # Show the 3 most ambiguous classifications
    if gaps:
        print("\n  Most ambiguous classifications (smallest gap):")
        paired = sorted(zip(gaps, accepted[:len(gaps)]), key=lambda x: x[0])
        for gap_val, r in paired[:3]:
            cands = r.get("top_candidates", [])
            top1  = (cands[0].get("hierarchy_path", "?") if len(cands) > 0 else "?")
            top2  = (cands[1].get("hierarchy_path", "?") if len(cands) > 1 else "?")
            issue = r.get("issue_title", r.get("issue", "?"))
            print(f"    gap={gap_val:.4f}  |  \"{top1[:60]}\"")
            print(f"                 vs \"{top2[:60]}\"")
            print(f"    Issue: {issue[:80]}")

    # ── Metric 9: Evidence Count Per Classification ────────────────────────────
    evidence_counts = [len(r.get("evidence", [])) for r in accepted]
    avg_evidence    = sum(evidence_counts) / len(evidence_counts) if evidence_counts else 0
    zero_evidence   = sum(1 for c in evidence_counts if c == 0)

    _sep("METRIC 9 — Evidence Count Per Classification")
    print(f"  Accepted classifications   : {len(accepted)}")
    print(f"  Avg evidence chunks        : {avg_evidence:.2f}")
    print(f"  Classifications w/ 0 chunks: {zero_evidence}")

    if avg_evidence > 1.5:
        m9_status = "GOOD"
        print("  STATUS → GOOD  (> 1.5 chunks avg — well grounded)")
    elif avg_evidence > 0.5:
        m9_status = "OK"
        print("  STATUS → ACCEPTABLE  (0.5–1.5, partially grounded)")
    else:
        m9_status = "BAD"
        print("  STATUS → BAD  (< 0.5 — classifications not grounded in docs)")

    # ── Metric 10: Node Coverage by Issues ────────────────────────────────────
    # Fix: divide by feature-only node count, not total (which includes chunks)
    matched_nodes = {r["node_id"] for r in accepted if r.get("node_id")}
    coverage_ratio = _div(len(matched_nodes), total_nodes)

    # Count how often each node won
    win_counter: dict[str, int] = {}
    for r in accepted:
        nid = r.get("node_id")
        if nid:
            win_counter[nid] = win_counter.get(nid, 0) + 1

    top_winner      = max(win_counter, key=win_counter.get) if win_counter else None
    top_winner_pct  = _div(win_counter[top_winner], len(accepted)) if top_winner else 0
    top_winner_name = (
        H.nodes[top_winner].get("label", top_winner)
        if (top_winner and top_winner in H.nodes)
        else (top_winner or "N/A")
    )

    _sep("METRIC 10 — Node Coverage by Issues")
    print(f"  Feature nodes total         : {total_nodes}")
    print(f"  Nodes matched by any issue  : {len(matched_nodes)}")
    print(f"  Coverage ratio              : {coverage_ratio:.1%}")
    print(f"  Top winning node            : '{top_winner_name}'")
    print(f"  Top node win share          : {top_winner_pct:.1%} of accepted issues")

    if top_winner_pct < 0.30:
        m10_status = "GOOD"
        print("  STATUS → GOOD  (no single node dominates)")
    elif top_winner_pct < 0.50:
        m10_status = "OK"
        print("  STATUS → ACCEPTABLE  (one node is popular but not dominating)")
    else:
        m10_status = "BAD"
        print("  STATUS → BAD  (> 50% issues land on one node — graph too coarse)")

    # Top 5 most-matched nodes
    print("\n  Top 5 most-matched nodes:")
    for nid, count in sorted(win_counter.items(), key=lambda x: -x[1])[:5]:
        name = (
            H.nodes[nid].get("label", nid)
            if nid in H.nodes else nid
        )
        pct = _div(count, len(accepted)) * 100
        print(f"    {count:>3} issues ({pct:4.1f}%)  →  {name}")

    # ── RAG Metrics Summary Table ──────────────────────────────────────────────
    print("\n")
    _sep("RAG METRICS SUMMARY")

    col_w = [35, 15, 8]
    rows  = [
        ["Metric",                       "Value",                    "Status"],
        ["Uncertain rate",               f"{uncertain_rate:.1%}",    m6_status],
        ["Avg confidence (accepted)",    f"{avg_conf:.4f}",          m7_status],
        ["Avg top-1 vs top-2 gap",       f"{avg_gap:.4f}",           m8_status],
        ["Avg evidence chunks",          f"{avg_evidence:.2f}",      m9_status],
        ["Node coverage ratio",          f"{coverage_ratio:.1%}",    "—"],
        ["Top node win share",           f"{top_winner_pct:.1%}",    m10_status],
    ]
    for row in rows:
        print(f"  {row[0]:<{col_w[0]}} {row[1]:<{col_w[1]}} {row[2]}")

    return {
        "uncertain_rate": uncertain_rate,
        "avg_conf":       avg_conf,
        "avg_gap":        avg_gap,
        "avg_evidence":   avg_evidence,
        "coverage_ratio": coverage_ratio,
        "top_winner_pct": top_winner_pct,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def run(
    graph_filename: str = "feature_graph.pkl",
    results_path:   str = None,
):
    """
    Load graph, compute graph metrics, and optionally compute RAG metrics.

    Args:
        graph_filename : .pkl filename inside GRAPH_DIR.
        results_path   : path to the classified .pkl file produced by
                         classify_issues.py.  If None, only graph metrics
                         are printed.
    """
    G = load_graph(graph_filename)

    compute_graph_metrics(G)

    if results_path:
        if not os.path.exists(results_path):
            raise FileNotFoundError(
                f"Results file not found: {results_path}\n"
                f"Run --classify first."
            )
        with open(results_path, "rb") as f:
            results = pickle.load(f)
        print(f"\nLoaded {len(results)} classified results from: {results_path}")
        compute_rag_metrics(results, G)
    else:
        # Auto-detect the classified pkl in ISSUES_DIR
        default_pkl = os.path.join(ISSUES_DIR, f"{DOCS_PROVIDER}_classified.pkl")
        if os.path.exists(default_pkl):
            print(f"\nAuto-detected results: {default_pkl}")
            with open(default_pkl, "rb") as f:
                results = pickle.load(f)
            print(f"Loaded {len(results)} classified results.")
            compute_rag_metrics(results, G)
        else:
            print(
                "\nNo results file provided and none found at default path.\n"
                "Run --classify to generate it, then re-run --metrics."
            )


if __name__ == "__main__":
    run()