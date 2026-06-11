"""
Phase 5 — RAG Compliance Agent
build_index.py

PURPOSE:
    Reads the two regulatory JSON knowledge bases (AMA for Athens, Loi Le Meur for Paris)
    and builds a persistent ChromaDB vector index under rag/chroma_db/.

    ChromaDB stores each article as a vector embedding so the compliance agent can
    retrieve the most relevant articles given a query about a specific listing.
    Think of it as a search engine that understands meaning, not just keywords.

HOW IT WORKS:
    1. Load both JSON files (AMA + Loi Le Meur)
    2. For each article, embed its text using a local SentenceTransformer model
       (all-MiniLM-L6-v2 — fast, free, runs locally, no API key needed)
    3. Store the embedding + full article metadata in ChromaDB
    4. The index is saved to disk at rag/chroma_db/ and persists between runs

RUN THIS ONCE before running compliance_agent.py or the notebook.
Re-run it if you update the JSON files.

NOTE: Do NOT commit rag/chroma_db/ to git — it is already in .gitignore.
"""

import json
import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ── Paths ──────────────────────────────────────────────────────────────────────
RAG_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(RAG_DIR, "chroma_db")
AMA_FILE = os.path.join(RAG_DIR, "regulations_ama.json")
LE_MEUR_FILE = os.path.join(RAG_DIR, "regulations_le_meur.json")

COLLECTION_NAME = "aria_regulations"


def load_regulations() -> list[dict]:
    """Load and merge both regulatory JSON files into a single list of articles."""
    with open(AMA_FILE, "r", encoding="utf-8") as f:
        ama = json.load(f)
    with open(LE_MEUR_FILE, "r", encoding="utf-8") as f:
        le_meur = json.load(f)
    all_articles = ama + le_meur
    print(f"Loaded {len(ama)} AMA articles (Athens) + {len(le_meur)} Loi Le Meur articles (Paris)")
    return all_articles


def build_index(articles: list[dict]) -> chromadb.Collection:
    """
    Create (or overwrite) the ChromaDB collection and populate it with all articles.

    Each article becomes one document in the collection:
    - document: the full article text (what gets embedded and searched)
    - metadata: all other fields (article_id, city, law, title, triggers, penalties)
      stored as structured data, retrieved alongside the text match
    - id: the article_id (unique identifier)
    """
    # SentenceTransformerEmbeddingFunction downloads the model on first run (~90MB),
    # then caches it locally. Subsequent runs are instant.
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # PersistentClient saves the index to disk automatically after every operation
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete and recreate the collection so re-runs start fresh (no duplicate articles)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}' — rebuilding from scratch")
    except Exception:
        pass  # Collection didn't exist yet — that's fine

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    # Prepare the three parallel lists ChromaDB expects
    documents = []   # the text that gets embedded
    metadatas = []   # structured metadata stored alongside each document
    ids = []         # unique string ID for each document

    for article in articles:
        # The document text is the full article text — this is what semantic search runs on
        documents.append(article["text"])

        # Metadata is everything else — returned with each search result
        # ChromaDB requires all metadata values to be str, int, float, or bool
        metadatas.append({
            "article_id":         article["article_id"],
            "city":               article["city"],
            "country":            article["country"],
            "law":                article["law"],
            "title":              article["title"],
            "compliance_trigger": article["compliance_trigger"],
            "compliance_check":   article["compliance_check"],
            "penalty":            article["penalty"],
        })

        ids.append(article["article_id"])

    # Add all articles to the collection in one batch call
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Indexed {len(ids)} articles into ChromaDB collection '{COLLECTION_NAME}'")
    print(f"Index saved to: {CHROMA_PATH}")
    return collection


def verify_index(collection: chromadb.Collection) -> None:
    """Quick sanity check: run a test query and print the top result."""
    print("\n── Verification query ──────────────────────────────────────────────")
    test_query = "unlicensed Athens listing without AADE registration number"
    results = collection.query(query_texts=[test_query], n_results=2)

    for i, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0])
    ):
        print(f"  Result {i+1}: [{meta['article_id']}] {meta['title']}")
        print(f"           City: {meta['city']} | Trigger: {meta['compliance_trigger']}")
        print(f"           Text preview: {doc[:120]}...")
    print("── Verification complete ───────────────────────────────────────────\n")


if __name__ == "__main__":
    print("=" * 60)
    print("ARIA Phase 5 — Building ChromaDB Regulation Index")
    print("=" * 60)

    articles = load_regulations()
    collection = build_index(articles)
    verify_index(collection)

    print("Index build complete. You can now run compliance_agent.py")
