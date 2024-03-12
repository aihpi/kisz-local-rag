import os
from utils import list_files, read_file, get_chunks
import chromadb
from chromadb.utils import embedding_functions
import requests, json, random
from parameters import EMBEDDING_MODEL, CHROMA_DATA_PATH
from parameters import LLMBASEURL, MODEL


def make_collection(data_path, collection_name, skip_included_files=True):
    """Create vector store collection from a set of documents"""

    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"},
    )

    files = list_files(data_path, extensions=('.txt', '.pdf'))
    print(f"Embedding files: {', '.join(files)} ...")

    if skip_included_files:
        sources = {m.get('source') for m in collection.get().get('metadatas')}

    for f in files:
        _, file_name = os.path.split(f)

        if skip_included_files and file_name in sources:
            print(file_name, "already in Vector-DB, skipping...")
            continue

        text = read_file(f)

        print(f"Getting chunks for {file_name} ...")
        chunks = get_chunks(text)

        print(f"Embedding and storing {file_name} ...")
        collection.add(
            documents=chunks,
            ids=[f"id{file_name[:-4]}.{j}" for j in range(len(chunks))],
            metadatas=[{"source": file_name, "part": n} for n in range(len(chunks))],
        )


def get_collection(vector_store_path, collection_name):
    """Load a saved vector store collection"""

    print(f"Loading collection {collection_name} ...")
    client = chromadb.PersistentClient(path=vector_store_path)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection = client.get_collection(name=collection_name, embedding_function=embedding_func)

    return collection


def get_relevant_text(collection, query='', nresults=2, sim_th=None):
    """Get relevant text from a collection for a given query"""

    query_result = collection.query(query_texts=query, n_results=nresults)
    docs = query_result.get('documents')[0]
    if sim_th:
        similarities = [1-d for d in query_result.get("distances")[0]]
        relevant_docs = [d for d, s in zip(docs, similarities) if s >= sim_th]
        relevant_text = ''.join(relevant_docs)
    else:
        relevant_text = ''.join(docs)
    return relevant_text


# LLM Funcs (Ollama)
def generate(prompt, top_k=5, top_p=0.9, temp=0.2):
    url = LLMBASEURL + "/generate"
    data = {
        "prompt": prompt,
        "model": MODEL,
        "stream": False,
        "options": {"temperature": temp, "top_p": top_p, "top_k": top_k},
    }

    try:
        r = requests.post(url, json=data)
        response_dic = json.loads(r.text)
        return response_dic.get('response', '')

    except Exception as e:
        print(e)


def llm_mockup(prompt, top_k=1, top_p=0.9, temp=0.5):
    return random.choice(["Yes!", "Not sure", "It depends", "42"])


def get_context_prompt(question, context):
    contextual_prompt = (
        "Use the following context to answer the question at the end. "
        "Keep the answer as concise as possible.\n"
        "Context:\n"
        f"{context}"
        "\nQuestion:\n"
        f"{question}"
    )

    return contextual_prompt


if __name__ == "__main__":
    # Quick RAG sample check
    from parameters import DATA_PATH, COLLECTION_NAME

    make_collection(DATA_PATH, COLLECTION_NAME)

    collection = get_collection(CHROMA_DATA_PATH, COLLECTION_NAME)

    # Query
    query = "Where can I learn about artificial intelligence in Berlin?"
    # query = "What happened to John McClane in Christmas?"
    # query = "Who is Sherlock Holmes?"

    print("\nQuery:", query)

    relevant_text = get_relevant_text(collection, query)

    print("\nRelevant text:")
    print(relevant_text)

    # LLM Cli
    print("\nQuering LLM...")
    context_query = get_context_prompt(query, relevant_text)
    bot_response = generate(context_query)

    print("\nModel Answer:")
    print(bot_response)
