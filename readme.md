# Code Documentation RAG with Syntax Understanding 🤖

An AI-powered chatbot that analyzes any GitHub repository, allowing you to ask questions and understand its code.

## 🌟 Overview

This project is a Retrieval-Augmented Generation (RAG) system designed to help developers and users understand a codebase without having to manually read through all the files. You provide a link to a public GitHub repository, and the application processes its contents—code files, documentation, and markdown—to build a searchable knowledge base. You can then ask questions in natural language about the repository's functionality, implementation details, or specific code snippets.

![App Screenshot](https://raw.githubusercontent.com/username/repo-name/main/path/to/your/screenshot.png)
*Note: You should replace the link above with a real screenshot of your running application.*

---

## ✨ Key Features

* **GitHub Repository Ingestion**: Clones any public GitHub repository to use as the data source.
* **Multi-Language Code Parsing**: Understands and processes various programming and markup languages (e.g., Python, JavaScript, Java, Markdown).
* **Conversational Q&A**: A user-friendly chat interface to ask questions about the codebase.
* **Context-Aware Responses**: Uses a RAG pipeline to retrieve relevant code snippets and documentation before generating a precise answer.
* **Fast and Efficient**: Powered by the fast Groq LPU™ Inference Engine for real-time responses.

---

## 🛠️ Tech Stack

* **Frontend**: [Streamlit](https://streamlit.io/)
* **LLM Framework**: [LangChain](https://www.langchain.com/)
* **LLM**: [Llama3 via Groq API](https://groq.com/)
* **Embeddings**: [Hugging Face Sentence Transformers](https://huggingface.co/sentence-transformers)
* **Vector Store**: [ChromaDB](https://www.trychroma.com/)
* **Deployment**: [Hugging Face Spaces](https://huggingface.co/spaces)

---

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

* Python 3.9+
* Git

### 1. Clone the Repository

First, clone this repository to your local machine:
```bash
git clone [https://github.com/your-username/code-rag-project.git](https://github.com/your-username/code-rag-project.git)
cd code-rag-project