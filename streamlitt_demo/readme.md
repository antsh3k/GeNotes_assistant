# 🧠 Build a Local RAG with Ollama

## ⚙️ Setup

### 1. Install Ollama and Models

Download and install [**Ollama**](https://ollama.com/), then pull the required models:

```bash
# Embedding model
EMBEDDING_MODEL="mxbai-embed-large"

# Chat model
CHAT_MODEL="llama3.2:3b"

# Provider
MODEL_PROVIDER="ollama"
```

These are configured in your .env file. Refer to .env.example for the required environment variables, and create your own .env based on it.

ℹ️ A list of data sources to be scraped can be found in requests.txt. These are all from Genotes but can be scraped from any website.

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
```

Activate it:

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 🚀 Running the Pipeline
From your terminal (e.g. VS Code terminal), execute the scripts in order:

```bash
python 1_scraping_data.py
python 2_chunking_embedding_ingestion.py
streamlit run 3_chatbot.py
```







