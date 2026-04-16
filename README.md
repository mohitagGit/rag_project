# 🏢 Varta Multi-Model RAG Application

A production-ready, multi-tenant Retrieval-Augmented Generation (RAG) application built with **Streamlit**, **LangChain**, and **ChromaDB**. This application allows users to upload secure documents and chat with them using dynamic model routing between **Groq (Llama 3.1)** and **Google (Gemini)**.

## ✨ Key Features

* **🧠 Dynamic Model Routing:** Seamlessly switch between AI engines (Groq Llama 3.1 8B, Google Gemini 2.5 Flash, Gemini 1.5) mid-conversation without needing to re-upload documents.
* **🔒 Anonymous Multi-Tenancy:** Implements session-based architecture. Every user gets a completely isolated, private vector database (`chroma_db_{session_id}`) and chat history memory bubble.
* **🗑️ Secure Data Governance:** Built-in "Delete My Data" functionality that safely breaks SQLite memory locks and physically destroys the user's specific vector data and chat history from the server.
* **💅 Enterprise UI/UX:** Features a locked-height, auto-scrolling chat container preventing the main page and headers from jumping around, mimicking true SaaS chat interfaces.
* **🛡️ Resilient Architecture:** Defensive programming implemented to handle API rate limits (HTTP 429) gracefully without crashing the UI.

## 🏗️ Folder Structure

```text
enterprise_rag_project/
├── .env                 # Secret API Keys (Google, Groq) - Ignored in Git
├── .gitignore           # Prevents secrets and data from hitting GitHub
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
│
├── app.py               # 🌐 Frontend Entry Point (Streamlit UI)
│
├── src/                 # ⚙️ Core Backend Logic
│   ├── __init__.py      
│   ├── rag_engine.py    # LLM routing and LangChain orchestration
│   ├── data_engine.py   # Document chunking, processing, and vectorization
│   └── prompts.py       # System prompt templates
│
└── data/                # 🗂️ Ephemeral Storage (Ignored in Git)
    ├── documents/       # Temporary uploaded PDFs
    └── chroma_db_*/     # Isolated user vector databases
