import os
import pickle
import networkx as nx
import chromadb
from sentence_transformers import SentenceTransformer

from src.config import (
    GRAPH_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    TOP_K,
)


def query_issue(issue_text: str, top_k: int = TOP_K):
    """
    Query the vector store and return ranked feature hierarchy matches.

    For each matching chunk:
      1. Embed the query (cosine space, normalised — matches build_graph).
      2. Retrieve top_k nearest chunks from Chroma.
      3. Walk the graph upward via predecessors to get the clean feature path.
      4. Report similarity score (0–1, higher = better).
    """

    if not issue_text.strip():
        raise ValueError("Issue text cannot be empty.")

    # ── Load graph ────────────────────────────────────────────────────────────
    graph_path = os.path.join(GRAPH_DIR, "feature_graph.pkl")
    if not os.path.exists(graph_path):
        raise FileNotFoundError("Feature graph not found. Run build() first.")

    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    # ── Load Chroma + model ───────────────────────────────────────────────────
    model         = SentenceTransformer(EMBED_MODEL)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection    = chroma_client.get_collection(name=COLLECTION_NAME)

    # ── Embed query — normalised to match build_graph cosine space ────────────
    query_embedding = model.encode(
        issue_text,
        normalize_embeddings=True,
    ).tolist()

    # ── Vector search ─────────────────────────────────────────────────────────
    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "distances", "documents"],
    )

    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]
    chunk_ids = raw.get("ids",       [[]])[0]

    print(f"\nQuery : {issue_text}")
    print("=" * 60)
    print(f"Top {top_k} matching features:\n")

    # ── Build results — traverse graph once per chunk ─────────────────────────
    results = []

    for rank, (chunk_id, metadata, distance) in enumerate(
        zip(chunk_ids, metadatas, distances), start=1
    ):
        # Cosine distance → similarity  (both in [0, 1] because cosine space)
        similarity = round(1.0 - distance, 4)

        # Walk graph upward — Issue 1+2 fix: use predecessor walk, not nx.ancestors
        graph_hierarchy = _walk_to_root(G, chunk_id)

        # Fallback: reconstruct from Chroma metadata if chunk not in graph
        if not graph_hierarchy:
            level_keys = sorted(
                [k for k in metadata
                 if k.startswith("level_") and not k.endswith("_depth")],
                key=lambda k: int(k.split("_")[1]),
            )
            graph_hierarchy = [metadata[k] for k in level_keys]

        print(f"Rank {rank}")
        print(f"  Similarity     : {similarity:.4f}")
        print(f"  Feature path   : {' → '.join(graph_hierarchy)}")
        print(f"  Source file    : {metadata.get('source_file', 'unknown')}")
        print(f"  Hierarchy depth: {len(graph_hierarchy)}")
        print()

        # Issue 4 fix: store computed hierarchy directly — no second traversal
        results.append({
            "chunk_id":       chunk_id,
            "similarity":     similarity,
            "hierarchy":      graph_hierarchy,
            "hierarchy_path": " → ".join(graph_hierarchy),
            "source_file":    metadata.get("source_file", ""),
        })

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _walk_to_root(G: nx.DiGraph, chunk_id: str) -> list:
    """
    Walk from a chunk node up to the root via predecessors and return
    the ordered list of feature labels (root → immediate parent).

    Issue 1 fix: predecessor walk guarantees root-first order on a tree/forest,
                 unlike nx.topological_sort which is not deterministically ordered.
    Issue 2 fix: O(depth) traversal instead of O(nodes) nx.ancestors BFS.
    """
    if chunk_id not in G:
        return []

    path = []
    current = chunk_id

    while True:
        parents = list(G.predecessors(current))

        if not parents:
            break   # reached root

        # In a tree each node has exactly one parent; take the first
        current = parents[0]
        node_data = G.nodes[current]

        if node_data.get("type") == "feature":
            # Use human-readable label stored by build_graph.py
            path.append(node_data.get("label", current))

    # path is built leaf→root, reverse to get root→leaf
    path.reverse()
    return path


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "example issue"
    query_issue(text)