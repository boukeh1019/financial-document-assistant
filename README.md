# Financial Document Assistant (RAG-based)

A Retrieval-Augmented Generation (RAG) application that enables question answering over financial and regulatory documents.  
The system is designed with **enterprise banking constraints** in mind: explainability, data privacy, and deployment flexibility.

---

## 🚀 What This Project Does

- Ingests financial and regulatory PDF documents
- Splits documents into semantically meaningful chunks
- Generates vector embeddings and stores them locally using FAISS
- Retrieves relevant document chunks at query time
- Uses a Large Language Model (LLM) to generate **grounded answers**
- Displays source documents to support transparency and auditability

---


## 📌 Current Status

✅ Local RAG pipeline implemented  
✅ Open-source LLM integration via Ollama  
🚧 Performance optimisation (GPU tuning, context limits)  
🚧 Cloud backend support (AWS Bedrock, Azure OpenAI)

---

## 🔮 Planned Enhancements

- AWS Bedrock integration
- Azure OpenAI integration
- Prompt engineering and guardrails
- Responsible AI checks (bias, uncertainty disclaimers)
- Improved latency and GPU utilisation
