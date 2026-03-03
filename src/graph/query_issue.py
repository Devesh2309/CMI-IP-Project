import os
import pickle
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
    """Query the vector store and return ranked feature hierarchy matches."""

    if not issue_text.strip():
        raise ValueError("Issue text cannot be empty.")

    model = SentenceTransformer(EMBED_MODEL)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    graph_path = os.path.join(GRAPH_DIR, "feature_graph.pkl")
    if not os.path.exists(graph_path):
        raise FileNotFoundError("Feature graph not found. Run build() first.")

    with open(graph_path, "rb") as f:
        _ = pickle.load(f)

    query_embedding = model.encode(issue_text).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    print("\nTop matching subfeatures:\n")

    for i, metadata in enumerate(metadatas):
        distance = distances[i] if i < len(distances) else None

        hierarchy = [
            metadata[key]
            for key in sorted(metadata.keys())
            if key.startswith("level_")
        ]

        print(f"Rank {i+1}")
        print("Hierarchy:", " → ".join(hierarchy))
        print("Distance:", distance)
        print()