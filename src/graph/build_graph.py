import os
import re
import uuid
import pickle
import networkx as nx
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb

from src.config import (
    MARKDOWN_DIR,
    GRAPH_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    BATCH_SIZE,
    RESET_COLLECTION,
)


def parse_markdown_hierarchy(md_text: str):
    """Parse markdown into hierarchical chunks."""
    lines = md_text.split("\n")
    current_hierarchy = []
    chunks = []
    buffer = []

    for line in lines:
        heading_match = re.match(r"^(#+)\s+(.*)", line)

        if heading_match:
            if buffer:
                chunks.append(
                    {
                        "hierarchy": current_hierarchy.copy(),
                        "content": "\n".join(buffer).strip(),
                    }
                )
                buffer = []

            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()

            current_hierarchy = current_hierarchy[: level - 1]
            current_hierarchy.append(title)
        else:
            buffer.append(line)

    if buffer:
        chunks.append(
            {
                "hierarchy": current_hierarchy.copy(),
                "content": "\n".join(buffer).strip(),
            }
        )

    return chunks


def build(
    markdown_dir: str = MARKDOWN_DIR,
    graph_dir: str = GRAPH_DIR,
    chroma_dir: str = CHROMA_DIR,
):
    """Build feature graph and populate Chroma vector store."""

    if not os.path.exists(markdown_dir):
        raise FileNotFoundError(f"Markdown directory not found: {markdown_dir}")

    os.makedirs(graph_dir, exist_ok=True)

    model = SentenceTransformer(EMBED_MODEL)
    chroma_client = chromadb.PersistentClient(path=chroma_dir)

    if RESET_COLLECTION:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:pass

    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    G = nx.DiGraph()

    documents_batch = []
    metadatas_batch = []
    ids_batch = []

    print("Processing markdown files...")

    for filename in tqdm(os.listdir(markdown_dir)):
        if not filename.endswith(".md"):
            continue

        file_path = os.path.join(markdown_dir, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            md_text = f.read()

        chunks = parse_markdown_hierarchy(md_text)

        for chunk in chunks:
            if len(chunk["content"]) < 50:
                continue

            chunk_id = str(uuid.uuid4())

            parent = None
            for level_name in chunk["hierarchy"]:
                if not G.has_node(level_name):
                    G.add_node(level_name, type="feature")

                if parent:
                    G.add_edge(parent, level_name)

                parent = level_name

            G.add_node(chunk_id, type="chunk")
            if parent:
                G.add_edge(parent, chunk_id)

            metadata = {
                "source_file": filename,
                "hierarchy_path": " → ".join(chunk["hierarchy"]),
            }

            for i, level in enumerate(chunk["hierarchy"]):
                metadata[f"level_{i+1}"] = level

            documents_batch.append(chunk["content"])
            metadatas_batch.append(metadata)
            ids_batch.append(chunk_id)

            if len(documents_batch) >= BATCH_SIZE:
                embeddings = model.encode(
                    documents_batch,
                    batch_size=BATCH_SIZE,
                    show_progress_bar=False,
                )

                collection.add(
                    ids=ids_batch,
                    embeddings=embeddings.tolist(),
                    documents=documents_batch,
                    metadatas=metadatas_batch,
                )

                documents_batch = []
                metadatas_batch = []
                ids_batch = []

    if documents_batch:
        embeddings = model.encode(
            documents_batch,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
        )

        collection.add(
            ids=ids_batch,
            embeddings=embeddings.tolist(),
            documents=documents_batch,
            metadatas=metadatas_batch,
        )

    graph_path = os.path.join(graph_dir, "feature_graph.pkl")
    with open(graph_path, "wb") as f:
        pickle.dump(G, f)

    print("Graph and Chroma DB saved.")
    return graph_path


if __name__ == "__main__":
    build()