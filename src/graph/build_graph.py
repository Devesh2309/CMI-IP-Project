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


def parse_markdown_hierarchy(md_text: str, filename: str):
    """
    Parse markdown into hierarchical chunks.

    Each chunk carries:
      - hierarchy : list of heading strings from root → leaf
      - content   : body text under that heading
      - levels    : corresponding heading depths (1 for H1, 2 for H2, …)

    Fix 1 (node collision): caller prefixes nodes with filename.
    Fix 2 (pre-heading content): content before the first heading is
           assigned to a synthetic root node derived from the filename.
    """
    lines = md_text.split("\n")

    # Fallback root label = filename without extension
    file_root = filename.replace(".md", "")

    current_hierarchy = [file_root]   # Fix 2: always start with a root
    current_levels    = [0]           # depth 0 = synthetic file root
    chunks = []
    buffer = []

    for line in lines:
        heading_match = re.match(r"^(#+)\s+(.*)", line)

        if heading_match:
            # Flush buffered content under the current heading
            if buffer:
                chunks.append({
                    "hierarchy": current_hierarchy.copy(),
                    "levels":    current_levels.copy(),
                    "content":   "\n".join(buffer).strip(),
                })
                buffer = []

            level = len(heading_match.group(1))   # 1 = H1, 2 = H2, …
            title = heading_match.group(2).strip()

            # Trim hierarchy back to the parent of this level
            # Keep entries whose depth is strictly less than `level`
            current_hierarchy = [h for h, l in zip(current_hierarchy, current_levels) if l < level]
            current_levels    = [l for l in current_levels if l < level]

            current_hierarchy.append(title)
            current_levels.append(level)

        else:
            buffer.append(line)

    # Final flush
    if buffer:
        chunks.append({
            "hierarchy": current_hierarchy.copy(),
            "levels":    current_levels.copy(),
            "content":   "\n".join(buffer).strip(),
        })

    return chunks


def build(
    markdown_dir: str = MARKDOWN_DIR,
    graph_dir:    str = GRAPH_DIR,
    chroma_dir:   str = CHROMA_DIR,
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
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # normalise to cosine so distances are in [0,1]
    )

    G = nx.DiGraph()

    documents_batch = []
    metadatas_batch = []
    ids_batch       = []

    print("Processing markdown files...")

    for filename in tqdm(sorted(os.listdir(markdown_dir))):
        if not filename.endswith(".md"):
            continue

        file_path = os.path.join(markdown_dir, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            md_text = f.read()

        chunks = parse_markdown_hierarchy(md_text, filename)

        for chunk in chunks:
            content = chunk["content"]

            # Skip chunks that are too short to be meaningful
            if len(content.strip()) < 50:
                continue

            chunk_id = str(uuid.uuid4())

            
            # Build graph nodes — Fix 1: prefix with filename to avoid
            # cross-document heading collisions (e.g. two files both having
            # an "Introduction" heading merging into the same node).
            
            parent = None
            for heading, depth in zip(chunk["hierarchy"], chunk["levels"]):
                # Unique node id = "filename::heading text"
                node_id = f"{filename}::{heading}"

                if not G.has_node(node_id):
                    G.add_node(
                        node_id,
                        type="feature",
                        label=heading,          # human-readable label
                        level=depth,            # Fix 3: store heading depth
                        source_file=filename,
                    )

                # Fix 4: only add edge if it doesn't already exist
                if parent and not G.has_edge(parent, node_id):
                    G.add_edge(parent, node_id)

                parent = node_id

            # Chunk node hangs off the deepest heading node
            G.add_node(chunk_id, type="chunk", source_file=filename)
            if parent and not G.has_edge(parent, chunk_id):
                G.add_edge(parent, chunk_id)

            
            # Metadata stored in Chroma
            
            hierarchy_path = " → ".join(chunk["hierarchy"])

            metadata = {
                "source_file":    filename,
                "hierarchy_path": hierarchy_path,
            }

            # Store each level separately so query_issue can reconstruct path
            for i, (heading, depth) in enumerate(zip(chunk["hierarchy"], chunk["levels"])):
                metadata[f"level_{i+1}"]       = heading
                metadata[f"level_{i+1}_depth"] = depth

            metadata["hierarchy_depth"] = len(chunk["hierarchy"])

            
            # Fix 5: enrich embedded text with hierarchy path so the
            # embedding carries feature-context, not just raw body text.
            
            enriched_text = f"{hierarchy_path}\n\n{content}"

            documents_batch.append(enriched_text)
            metadatas_batch.append(metadata)
            ids_batch.append(chunk_id)

            if len(documents_batch) >= BATCH_SIZE:
                _flush_batch(model, collection, documents_batch, metadatas_batch, ids_batch)
                documents_batch = []
                metadatas_batch = []
                ids_batch       = []

    # Final partial batch
    if documents_batch:
        _flush_batch(model, collection, documents_batch, metadatas_batch, ids_batch)

    # Persist graph
    graph_path = os.path.join(graph_dir, "feature_graph.pkl")
    with open(graph_path, "wb") as f:
        pickle.dump(G, f)

    n_features = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "feature")
    n_chunks   = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "chunk")

    print(f"\nGraph saved   : {graph_path}")
    print(f"  Feature nodes : {n_features}")
    print(f"  Chunk nodes   : {n_chunks}")
    print(f"  Edges         : {G.number_of_edges()}")
    print(f"Chroma saved  : {chroma_dir}  ({collection.count()} vectors)")

    return graph_path


# Internal helper
def _flush_batch(model, collection, documents, metadatas, ids):
    """Encode a batch and upsert into Chroma."""
    embeddings = model.encode(
        documents,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,   # required for cosine space
    )
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )


if __name__ == "__main__":
    build()