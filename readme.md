<!-- Nothing to show

for emediing generation the models that we can use are

sentence-transformers/all-MiniLM-L6-v2 (fast, small)
sentence-transformers/all-mpnet-base-v2 (high quality)
BAAI/bge-base-en-v1.5 (very strong for search) -->

# YouTube RAG Chatbot

A chatbot that answers questions about YouTube videos using RAG (Retrieval Augmented Generation).

## Features

- Extracts transcripts from YouTube videos
- Uses FAISS for vector storage
- Powered by HuggingFace models

## Installation

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/youtube-chatbot.git
cd youtube-chatbot
```

2. Create virtual environment:

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create `.env` file and add your HuggingFace token:

```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

## Usage

```bash
python youtube_chatboat.py
```

## Requirements

- Python 3.8+
- HuggingFace API token
