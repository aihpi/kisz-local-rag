#!/usr/bin/python3

import gradio as gr
import requests, json
import random

# $ ollama serve
BASEURL = "http://localhost:11434/api"
MODEL = 'mistral'


def generate(prompt, context, top_k=5, top_p=0.9, temp=0.5):
    url = BASEURL + "/generate"
    data = {
        "prompt": prompt,
        "model": MODEL,
        "stream": False,
        "context": context,
        "options": {"temperature": temp, "top_p": top_p, "top_k": top_k},
    }

    try:
        r = requests.post(url, json=data)
        response_dic = json.loads(r.text)
        return response_dic.get('response', ''), response_dic.get('context', '')

    except Exception as e:
        print(e)


def llm_mockup(prompt, top_k=1, top_p=0.9, temp=0.5, context=[]):
    return random.choice(["Yes!", "Not sure", "It depends", "42"]), context


def chat(user_msg, history, top_k, top_p, temp, context=[]):
    bot_response, context[:] = generate(
        user_msg, context=context, top_k=top_k, top_p=top_p, temp=temp
    )
    # bot_response, _ = llm_mockup(user_msg, top_k=top_k, top_p=top_p, temp=temp)
    return bot_response


chatbot = gr.ChatInterface(
    chat,
    title=f"Your Local GPT. ({MODEL})",
    chatbot=gr.Chatbot(height=700),
    additional_inputs=[
        gr.Slider(1, 10, value=5, step=1, label="Top K"),
        gr.Slider(0.1, 1, value=0.9, step=0.1, label="Top p"),
        gr.Slider(0.1, 1, value=0.5, step=0.1, label="Temp"),
    ],
)

if __name__ == '__main__':
    chatbot.launch(inbrowser=True)
