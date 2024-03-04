from ragfuncs import (
    make_collection,
    get_collection,
    get_relevant_text,
    get_context_prompt,
    generate,
)
from parameters import DATA_PATH
from parameters import CHROMA_DATA_PATH, COLLECTION_NAME, MODEL


def main():
    make_collection(DATA_PATH, COLLECTION_NAME)
    collection = get_collection(CHROMA_DATA_PATH, COLLECTION_NAME)

    print(f"\n============== Local RAG (Model: {MODEL}) ==============")
    print("(Press 'q' to quit)")
    while True:
        user_input = input("\nYour prompt: ")
        if user_input == "q":
            break

        relevant_text = get_relevant_text(collection, user_input)

        # LLM Cli
        context_query = get_context_prompt(user_input, relevant_text)
        rag_response = generate(context_query, top_k=3, top_p=0.9, temp=0.3)

        print("Answer:")
        print(rag_response)


if __name__ == '__main__':
    main()
