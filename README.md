# CMI-IP-Project
this is a repository to store and run code for my Chennai Mathematical Institute Industry Project 

# API Feature Knowledge Graph Builder (CMI Industry Project)

This project builds a Feature–Subfeature Knowledge Graph from API documentation
and enables RAG-based issue classification.

## Pipeline

1. Discover API Links
2. Render Pages (Playwright)
3. Convert HTML → Markdown
4. Build Feature Hierarchy Graph
5. Embed & Store in Chroma
6. Query Issues

## Setup

pip install -r requirements.txt
playwright install

Create .env with:
FIRECRAWL_API_KEY=...

## Run

python main.py --discover
python main.py --render
python main.py --markdown
python main.py --build

Query:
python main.py --query "Payment intent failing"