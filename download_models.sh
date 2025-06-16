#!/bin/bash
set -e

echo "Waiting for Ollama to be ready..."
until /usr/bin/curl -s http://ollama:11434/api/tags > /dev/null; do
  echo "Ollama is not ready yet. Waiting..."
  sleep 5
done

echo "Downloading models..."

# Download models using the Ollama API https://ollama.com/library for list of models
for model in "nomic-embed-text" "mxbai-embed-large" "llama3.2:3b"; do
  echo "Pulling $model..."
  /usr/bin/curl -X POST http://ollama:11434/api/pull -d "{\"name\": \"$model\"}"
  echo -e "\nFinished pulling $model\n"
done

echo "All models downloaded successfully!"
