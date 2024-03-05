from ragfuncs import (
    make_collection,
    get_collection,
    get_relevant_text,
    get_context_prompt,
    generate,
)
from parameters import DATA_PATH
from parameters import CHROMA_DATA_PATH, COLLECTION_NAME, GUI_TITLE
import gradio as gr


def main():
    make_collection(DATA_PATH, COLLECTION_NAME)
    collection = get_collection(CHROMA_DATA_PATH, COLLECTION_NAME)

    def rag(user_msg, history, top_k, top_p, temp):
        relevant_text = get_relevant_text(collection, user_msg, sim_th=0.4)
        context_query = get_context_prompt(user_msg, relevant_text)
        bot_response = generate(context_query, top_k=top_k, top_p=top_p, temp=temp)
        return bot_response

    chatgui = gr.ChatInterface(
        rag,
        title=GUI_TITLE,
        chatbot=gr.Chatbot(height=700),
        additional_inputs=[
            gr.Slider(1, 10, value=5, step=1, label="Top k"),
            gr.Slider(0.1, 1, value=0.9, step=0.1, label="Top p"),
            gr.Slider(0.1, 1, value=0.5, step=0.1, label="Temp"),
        ],
    )
    chatgui.launch()


if __name__ == '__main__':
    main()
