#!/usr/bin/python3
# Local gpt with ollama
import requests, json, random

# $ ollama serve
BASEURL = f"http://localhost:11434/api"
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


def llm_mockup(prompt, context=[], top_k=1, top_p=0.9, temp=0.5):
    # TODO: complete to depic the the use of top_k, top_p and temp?
    return random.choice(["Yes!", "Not sure", "It depens", "42"]), context


if __name__ == '__main__':
    user_input = "Hi. who are you?"
    ollama_context = []
    print(f"Start chatting with {MODEL} model (Press q to quit)\n")
    while user_input != "q":
        print("Context length:", len(ollama_context))
        bot_response, ollama_context = generate(
            user_input, context=ollama_context, top_k=10, top_p=0.9, temp=0.5
        )
        # bot_response, _ = llm_mockup(user_input, top_k=10, top_p=0.9, temp=0.5)
        print("Model message:")
        print(bot_response)
        user_input = input("\nYour prompt: ")
