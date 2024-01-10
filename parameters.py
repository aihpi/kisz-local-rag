# General configuration parameters on each area

# Load text/documents
DATA_PATH = "sample_data/"

# Text embedding (sentence transformer)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Vector Store
CHROMA_DATA_PATH = "chroma_data/"
COLLECTION_NAME = "sample_docs"

# LLM (ollama)
LLMBASEURL = f"http://localhost:11434/api"
MODEL = 'mistral'

# Frontend
GUI_TITLE = f"Local RAG System ({MODEL})"
