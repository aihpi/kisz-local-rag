import chromadb
from chromadb.utils import embedding_functions

from parameters import EMBEDDING_MODEL, CHROMA_DATA_PATH, COLLECTION_NAME

# Load and use stored DB
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)

# Searching on the Database
query1 = "Where can I learn about artificial intelligence in Berlin?"
query2 = "What happend in christmas with John McClane?"
query3 = "Who is Watson?"

queries = [query1, query2, query3]

query_results = collection.query(
    query_texts=queries,
    n_results=2,
)
# print("Query dict keys:")
# print(query_results.keys())

print("Quering collection:", COLLECTION_NAME)

for i in range(len(queries)):
    print("\nQuery:", queries[i])

    print("\nResults:")
    for j in range(len(query_results["ids"][i])):
        print("\nid:", query_results["ids"][i][j])
        print("Text:")
        print(query_results["documents"][i][j])
        print("Distance:", round(query_results["distances"][i][j], 2))
        print("Metadata:", query_results["metadatas"][i][j])

    print(80 * "-")
