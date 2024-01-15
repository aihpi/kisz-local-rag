#!/bin/bash
echo 'Installing/Updating ollama...' && \
curl https://ollama.ai/install.sh | sh && \
echo 'Pulling model...' && \
ollama pull mistral && \
echo 'Creating virtual environment...' && \
python3 -m venv .myvenv && \
echo 'Activating virtual environment...' && \
source .myvenv/bin/activate && \
echo 'Installing requirements...' && \
pip3 install -r requirements.txt
