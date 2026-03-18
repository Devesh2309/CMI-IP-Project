import os
import pickle
from pyvis.network import Network

from src.config import GRAPH_DIR


def visualize(
    graph_filename: str = "feature_graph.pkl",
    output_filename: str = "feature_graph_visualization.html",
    show_chunks: bool = False,
):
    """Visualize the feature graph using PyVis."""

    graph_path = os.path.join(GRAPH_DIR, graph_filename)
    output_path = os.path.join(GRAPH_DIR, output_filename)

    if not os.path.exists(graph_path):
        raise FileNotFoundError("Feature graph not found. Run build() first.")

    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    net = Network(
        height="800px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="black",
    )

    net.barnes_hut()

    for node, data in G.nodes(data=True):
        node_type = data.get("type", "feature")

        if node_type == "chunk" and not show_chunks:
            continue

        if node_type == "feature":
            net.add_node(node, label=node, color="#1f77b4", size=20)

        elif node_type == "chunk":
            net.add_node(node, label="chunk", color="#ff7f0e", size=6)

    for source, target in G.edges():
        if not show_chunks:
            if G.nodes[source].get("type") == "chunk":
                continue
            if G.nodes[target].get("type") == "chunk":
                continue

        net.add_edge(source, target)

    net.write_html(output_path)
    print(f"Graph saved to: {output_path}")


if __name__ == "__main__":
    visualize(show_chunks=True)