# Local-RAG-System-Python 🚀

A privacy-first, fully open-source **Retrieval-Augmented Generation (RAG)** system built using Python. This project runs entirely on your local machine, eliminating API subscription costs and ensuring absolute data security. It allows users to upload custom PDF documents and ask questions directly to a local Large Language Model (LLM) using semantic contextual search.

---

## 🏗️ System Architecture

This project is built as a **Naive RAG** system and is divided into two operational pipelines:

### 1. The Ingestion Pipeline (Offline)
*   **Document Loading:** Parses raw PDF text pages seamlessly.
*   **Text Chunking:** Slices lengthy documents using `RecursiveCharacterTextSplitter` into overlapping blocks to preserve localized structural context.
*   **Vector Embeddings:** Automatically translates text blocks into mathematical representations using the open-source `all-MiniLM-L6-v2` Sentence Transformer.
*   **Vector Store:** Vector indices are compiled and saved locally on disk inside a structured `Chroma` database.

### 2. Retrieval & Generation Pipeline (Runtime)
*   **Query Embedding:** Conversational queries are processed using the matching embedding framework.
*   **Vector Search:** Executes a Cosine Similarity lookup across the local database vector space to pull the top matching content chunks.
*   **Augmentation & Inference:** Compiles the retrieved context chunks with the user prompt and passes them to a localized `Llama 3` instance powered by `Ollama` for a accurate, hallucination-free response.

---

## 📊 Pipeline Flowchart

```text

===========================================================================================
                                   THE RAG ARCHITECTURE
===========================================================================================

1. THE INGESTION PIPELINE (Offline Process)
---------------+        +------------------+        +----------------------+        +------------+
|  Source PDF  | -----> | Text Splitter    | -----> | Embedding Model      | -----> | Local      |
|  Document    |        | (Recursive 500ch)|        | (all-MiniLM-L6-v2)   |        | ChromaDB   |
---------------+        +------------------+        +----------------------+        +------------+
                                                                                          |
                                                                                          |
2. THE RETRIEVAL & GENERATION PIPELINE (Real-Time Runtime)                                |
                                                                                          v
+--------------+        +------------------+        +----------------------+        +------------+
|  User Query  | -----> | Embedding Model  | -----> | Vector Search        | -----> | Retrieved  |
|  (Question)  |        | (all-MiniLM-L6-v2)        | (Cosine Similarity)  |        | Chunks     |
+--------------+        +------------------+        +----------------------+        +------------+
                                                                                          |
                                                                                          v
                        +------------------+        +----------------------+        +------------+
                        | Generated Answer | <----- | Local LLM            | <----- | Augmented  |
                        | (User Output)    |        | (Ollama / Llama 3)   |        | Prompt     |
                        +------------------+        +----------------------+        +------------+
```

🛠️ Tech Stack & Tooling
Orchestration Framework: LangChain & LangChain-Classic (Python)

Vector Database: ChromaDB (Local Directory Storage)

Embedding Model: all-MiniLM-L6-v2 (Via HuggingFace/SentenceTransformers)

Local LLM Engine: Ollama (Llama 3)

🚀 Step-by-Step Setup Guide
1. Prerequisites
Ensure you have Python 3.9+ and Ollama installed on your system.

Download Ollama from ollama.com

Pull the Llama 3 model weights in your terminal:

Bash
ollama run llama3


### 2. Clone and Install Dependencies
Clone this repository to your workspace and install the required modules:
bash
git clone [https://github.com/YOUR_USERNAME/Local-RAG-System-Python.git](https://github.com/YOUR_USERNAME/Local-RAG-System-Python.git)
cd Local-RAG-System-Python
pip install langchain langchain-community langchain-classic sentence-transformers chromadb pypdf langchain-ollama
3. Run the Ingestion Pipeline
Place your target PDF (e.g., sample_policy.pdf) in the project directory, update the file name inside ingest.py, and run the extraction and embedding step:

#### Bash
python ingest.py
This updates your local workspace folder with a fresh ./chroma_db index.

4. Query Your Document
Ensure Ollama is running actively in the background. Enter your specific query in query.py and run the generation loop:

#### Bash
python query.py
