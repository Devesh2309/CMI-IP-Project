import os
import pickle
from collections import Counter
import pandas as pd
import networkx as nx

# 🔗 Import YOUR config (important)
from src.config import GRAPH_DIR


# ============================================================
# Helpers
# ============================================================

def load_graph(graph_filename="feature_graph.pkl"):
    graph_path = os.path.join(GRAPH_DIR, graph_filename)

    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph not found at {graph_path}")

    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    print(f"Loaded graph from: {graph_path}")
    return G


def get_feature_subgraph(G):
    """Remove chunk nodes — keep only feature hierarchy."""
    return G.subgraph(
        [n for n, d in G.nodes(data=True) if d.get("type") == "feature"]
    ).copy()


def compute_root_metrics(G, roots):
    """Compute subtree size + depth per root."""

    def dfs(root):
        visited = set()
        stack = [(root, 0)]
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

    return {r: dfs(r) for r in roots}


def build_nodes_df(G):
    """Build DataFrame similar to your colleague's expectation."""
    rows = []

    for node, attr in G.nodes(data=True):
        if attr.get("type") != "feature":
            continue

        rows.append({
            "node": node,
            "name": node,
            "level": attr.get("level", None),  # not stored currently
            "support_count": attr.get("support_count", 1)  # default fallback
        })

    return pd.DataFrame(rows)


# ============================================================
# GRAPH METRICS (1–5)
# ============================================================

def compute_graph_metrics(G):

    print("\n" + "=" * 55)
    print("  GRAPH METRICS")
    print("=" * 55)

    # 👉 Only feature nodes
    H = get_feature_subgraph(G)

    total_nodes = H.number_of_nodes()
    roots = [n for n in H.nodes() if H.in_degree(n) == 0]

    # ── Metric 1: Root Ratio ─────────────────────────────
    root_ratio = len(roots) / total_nodes if total_nodes else 0

    print("\nMETRIC 1 — Root Ratio")
    print(f"  Total nodes : {total_nodes}")
    print(f"  Root nodes  : {len(roots)}")
    print(f"  Root ratio  : {root_ratio:.1%}")

    # ── Metric 2 + 3: Subtree + Depth ────────────────────
    root_metrics = compute_root_metrics(H, roots)
    subtree_sizes = [root_metrics[r][0] for r in roots]

    if subtree_sizes:
        max_subtree = max(subtree_sizes)
        min_subtree = min(subtree_sizes)
        avg_subtree = sum(subtree_sizes) / len(subtree_sizes)
        giant_share = max_subtree / total_nodes
    else:
        max_subtree = min_subtree = avg_subtree = giant_share = 0

    print("\nMETRIC 2 — Subtree Size Distribution")
    print(f"  Avg subtree size : {avg_subtree:.2f}")
    print(f"  Largest subtree  : {max_subtree}")
    print(f"  Smallest subtree : {min_subtree}")
    print(f"  Largest share    : {giant_share:.1%}")

    # Depth
    depths = [root_metrics[r][1] for r in roots]

    if depths:
        max_depth = max(depths)
        avg_depth = sum(depths) / len(depths)
        min_depth = min(depths)
    else:
        max_depth = avg_depth = min_depth = 0

    print("\nMETRIC 3 — Depth")
    print(f"  Max depth : {max_depth}")
    print(f"  Avg depth : {avg_depth:.2f}")
    print(f"  Min depth : {min_depth}")

    print("\n  Depth distribution:")
    depth_hist = Counter(depths)
    for d in sorted(depth_hist):
        bar = "█" * depth_hist[d]
        print(f"    depth {d} | {bar} ({depth_hist[d]})")

    # ── Metric 4: Support Count ──────────────────────────
    nodes_df = build_nodes_df(H)

    if not nodes_df.empty:
        single_occ = (nodes_df["support_count"] == 1).sum()
        single_ratio = single_occ / len(nodes_df)

        print("\nMETRIC 4 — Support Count")
        print(f"  Nodes seen once : {single_occ} ({single_ratio:.1%})")
        print(f"  Avg support     : {nodes_df['support_count'].mean():.2f}")
    else:
        print("\nMETRIC 4 — Support Count (No data)")

    # ── Metric 5: Heading Levels ─────────────────────────
    if nodes_df["level"].notna().any():
        level_counts = nodes_df["level"].value_counts().sort_index()

        print("\nMETRIC 5 — Heading Levels")
        for lvl, cnt in level_counts.items():
            print(f"  H{lvl}: {cnt}")
    else:
        print("\nMETRIC 5 — Heading Levels (Not available)")


# ============================================================
# RAG METRICS (6–10)
# ============================================================

def compute_rag_metrics(results, G):

    print("\n" + "=" * 55)
    print("  RAG METRICS")
    print("=" * 55)

    total = len(results)
    uncertain = sum(1 for r in results if r["result"] == "Uncertain")
    uncertain_rate = uncertain / total if total else 0

    print("\nMETRIC 6 — Uncertain Rate")
    print(f"  {uncertain_rate:.1%}")

    accepted = [r for r in results if r["result"] != "Uncertain"]

    # Metric 7
    avg_conf = (
        sum(r["confidence_score"] for r in accepted) / len(accepted)
        if accepted else 0
    )

    print("\nMETRIC 7 — Avg Confidence")
    print(f"  {avg_conf:.4f}")

    # Metric 8
    gaps = []
    for r in accepted:
        cands = r.get("top_candidates", [])
        if len(cands) >= 2:
            gaps.append(cands[0]["score"] - cands[1]["score"])

    avg_gap = sum(gaps) / len(gaps) if gaps else 0

    print("\nMETRIC 8 — Top-1 vs Top-2 Gap")
    print(f"  {avg_gap:.4f}")

    # Metric 9
    evidence_counts = [len(r.get("evidence", [])) for r in accepted]
    avg_evidence = sum(evidence_counts) / len(evidence_counts) if evidence_counts else 0

    print("\nMETRIC 9 — Evidence Count")
    print(f"  {avg_evidence:.2f}")

    # Metric 10
    matched_nodes = {r["node_id"] for r in accepted if "node_id" in r}
    coverage = len(matched_nodes) / G.number_of_nodes()

    print("\nMETRIC 10 — Node Coverage")
    print(f"  {coverage:.1%}")


# ============================================================
# MAIN
# ============================================================

def run(graph_filename="feature_graph.pkl", results_path=None):

    G = load_graph(graph_filename)

    compute_graph_metrics(G)

    if results_path:
        if not os.path.exists(results_path):
            raise FileNotFoundError("Results file not found")

        with open(results_path, "rb") as f:
            results = pickle.load(f)

        compute_rag_metrics(results, G)


if __name__ == "__main__":
    run()