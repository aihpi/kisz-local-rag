from os import getenv
from urllib.parse import urljoin

# General configuration parameters on each area

# Load text/documents
DATA_PATH = "sample_data/"

# Text embedding (sentence transformer)
EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"

# Vector Store
CHROMA_DATA_PATH = "chroma_data/"
COLLECTION_NAME = "sample_docs"

# LLM (ollama)
LLMBASEURL = urljoin(getenv("OLLAMA_HOST", "http://localhost:11434"), "api")
MODEL = "phi3"

# Frontend
GUI_TITLE = f"Local RAG System ({MODEL})"
