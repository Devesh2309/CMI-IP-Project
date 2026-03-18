# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --------- API KEYS ----------
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

# --------- BASE URL ----------
BASE_URL = "https://support.discord.com/hc/en-us"

# --------- DATA DIRECTORIES ----------
BASE_DATA_DIR = "data"
DOCS_PROVIDER = os.getenv("DOCS_PROVIDER")

MARKDOWN_DIR = os.path.join(BASE_DATA_DIR, "markdown", f"{DOCS_PROVIDER}_md")
HTML_DIR = os.path.join(BASE_DATA_DIR, "rendered_html", f"{DOCS_PROVIDER}_html")
LINKS_FILE = os.path.join(BASE_DATA_DIR, "links", f"{DOCS_PROVIDER}_api_links.txt")
LINKS_DIR = os.path.join(BASE_DATA_DIR, "links")
ISSUES_DIR = os.path.join(BASE_DATA_DIR, "issues")

GRAPH_DIR = os.path.join(BASE_DATA_DIR, f"{DOCS_PROVIDER}", "graph")
CHROMA_DIR = os.path.join(BASE_DATA_DIR, f"{DOCS_PROVIDER}", "_chroma_db")

# --------- MODEL CONFIG ----------
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "api_features"
BATCH_SIZE = 32
TOP_K = 5
RESET_COLLECTION = False