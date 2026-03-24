import os
import pickle
from pyvis.network import Network

from src.config import GRAPH_DIR


# Colour and size per heading depth
# depth 0 = synthetic file-root, 1 = H1, 2 = H2, etc.
_DEPTH_COLOUR = {
    0: "#2c3e50",   # dark slate  — file root
    1: "#2980b9",   # strong blue — H1
    2: "#27ae60",   # green       — H2
    3: "#f39c12",   # amber       — H3
    4: "#8e44ad",   # purple      — H4
}
_DEFAULT_COLOUR = "#95a5a6"   # grey for H5+

_DEPTH_SIZE = {
    0: 35,
    1: 28,
    2: 22,
    3: 16,
    4: 12,
}
_DEFAULT_SIZE = 9


def visualize(
    graph_filename:  str  = "feature_graph.pkl",
    output_filename: str  = "feature_graph_visualization.html",
    output_dir:      str  = None,
    show_chunks:     bool = False,
):
    """
    Visualize the feature graph using PyVis.

    Args:
        graph_filename:  Name of the .pkl graph file inside GRAPH_DIR.
        output_filename: Name of the output HTML file.
        output_dir:      Where to save the HTML. Defaults to GRAPH_DIR.
        show_chunks:     Whether to render chunk (UUID) nodes.
    """

    graph_path  = os.path.join(GRAPH_DIR, graph_filename)
    output_dir  = output_dir or GRAPH_DIR
    output_path = os.path.join(output_dir, output_filename)

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(graph_path):
        raise FileNotFoundError("Feature graph not found. Run build() first.")

    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    net = Network(
        height="900px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#2c3e50",
    )

    # Tuned physics — prevents hairball on large graphs
    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=150,
        spring_strength=0.05,
        damping=0.09,
    )

    # ── Track which nodes were actually added so edges stay consistent ────────
    added_nodes = set()

    for node, data in G.nodes(data=True):
        node_type = data.get("type", "feature")

        if node_type == "chunk":
            if not show_chunks:
                continue
            # Chunk nodes: small, orange, tooltip shows source file
            tooltip = f"chunk\nsource: {data.get('source_file', '?')}"
            net.add_node(
                node,
                label="●",
                color="#ff7f0e",
                size=5,
                title=tooltip,
                shape="dot",
            )
            added_nodes.add(node)

        elif node_type == "feature":
            # Fix 1: use stored human-readable label, not raw node ID
            label  = data.get("label", node)
            depth  = data.get("level", 0)
            source = data.get("source_file", "")

            colour = _DEPTH_COLOUR.get(depth, _DEFAULT_COLOUR)
            size   = _DEPTH_SIZE.get(depth, _DEFAULT_SIZE)

            # Fix 3: rich hover tooltip
            tooltip = (
                f"{label}\n"
                f"Depth : H{depth} {'(file root)' if depth == 0 else ''}\n"
                f"Source: {source}\n"
                f"Node  : {node}"
            )

            net.add_node(
                node,
                label=label,        # Fix 1: clean heading text only
                color=colour,       # Fix 4: colour by depth
                size=size,          # Fix 4: size by depth
                title=tooltip,      # Fix 3: hover info
                font={"size": max(8, 18 - depth * 2)},
            )
            added_nodes.add(node)

    # ── Only add edges where both endpoints were added ─────────────────────
    # Fix 2: use added_nodes set instead of re-checking node type on the fly
    for source, target in G.edges():
        if source in added_nodes and target in added_nodes:
            net.add_edge(source, target, arrows="to", color="#232323")

    net.write_html(output_path)
    print(f"Visualization saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    visualize(show_chunks=False)