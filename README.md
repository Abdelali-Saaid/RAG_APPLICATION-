# Modular RAG Pro: Advanced Document Intelligence

Modular RAG Pro is a high-performance Retrieval-Augmented Generation (RAG) system built with a modular architecture. It leverages state-of-the-art LLMs (via Groq) and local embedding models to provide accurate, faithful, and relevant answers from your private document library.

---

## Visual Demo
Watching the system in action:

<video src="Implementation.mp4" controls="controls" style="max-width: 100%;">
  Your browser does not support the video tag.
</video>

> [!NOTE]
> The demo is provided as an `Implementation.mp4` file.

---

## Key Features
- **Hybrid Retrieval**: Combines Vector Search (FAISS) with BM25 for maximum accuracy.
- **Ultra-Fast Inference**: Integrated with Groq (Llama-3.3-70b-versatile) for near-instant responses.
- **Built-in Evaluation**: Real-time "Faithfulness" and "Relevancy" scoring for every answer.
- **Modular Design**: Clean separation between ingestion, indexing, querying, and storage.
- **Chat Archive**: Persistent session management allowing you to save and resume conversations.
- **Modern UI**: Professional Streamlit interface with dynamic Lucide icons and theme-aware styling.

---

## Project Structure
```text
RAG_APPLICATION/
├── src/                # Core Logic
│   ├── loader.py       # Multi-format document ingestion
│   ├── indexer.py      # Vector/BM25 index orchestration
│   ├── query_engine.py # Retrieval & generation logic
│   ├── storage.py      # Local persistence for FAISS/metadata
│   ├── evaluator.py    # Faithfulness/Relevancy components
│   └── history_manager.py # Chat persistence logic
├── data/               # Raw documents (PDF, TXT, MD, etc.)
├── storage/            # Persisted index files
├── history/            # Saved chat sessions
├── app.py              # Streamlit Web Interface
└── requirements.txt    # Project dependencies
```

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- A [Groq API Key](https://console.groq.com/)

### 2. Installation
```bash
# Clone the repository (if applicable)
# git clone <your-repo-url>
# cd RAG_APPLICATION

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the Application
```bash
streamlit run app.py
```

---

## Configuration Details
The system automatically detects files in the `data/` directory. You can configure:
- **Allowed Extensions**: PDF, TXT, MD, CSV, DOCX via the UI sidebar.
- **Embedding Model**: Uses `all-MiniLM-L6-v2` locally via HuggingFace for speed and privacy.

---

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
