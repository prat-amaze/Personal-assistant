# 🤖 Personal AI Assistant (RAG + LangGraph)

An AI-powered personal assistant that answers questions about me using my **resume and LinkedIn data**.

Built with **LangGraph + Retrieval-Augmented Generation (RAG)**, this project demonstrates how to create a **context-aware AI agent** with a clean, interactive UI.

---

## 🚀 Demo

Ask questions like:
- What projects have you built?
- What are your technical skills?
- Tell me about your work experience
- What roles are you open to?

---

## 🧠 How It Works

### 1. Data Ingestion
- Resume + LinkedIn parsed into chunks  
- Stored in **ChromaDB** with metadata (`projects`, `skills`, `experience`, etc.)

### 2. Query Flow (LangGraph)
- **Classifier Node** → identifies query type  
- **Retriever Node** → fetches relevant chunks using metadata filter  
- **Answer Node** → generates response using LLM  

### 3. Frontend
- Minimal SaaS-style UI  
- Typing animation + quick suggestion pills  

---

## 🏗️ Tech Stack

- **Backend**: FastAPI  
- **Orchestration**: LangGraph  
- **LLM**: Google Gemini (`langchain-google-genai`)  
- **Vector DB**: ChromaDB  
- **Frontend**: HTML, CSS, JavaScript  
- **Deployment**: Railway  

---

## 📁 Project Structure

```
.
├── main.py
├── graph.py
├── chroma_db/
├── static/
│   ├── index.html
│   └── profile.jpg
├── requirements.txt
├── Procfile
└── README.md
```

---

## ⚙️ Setup & Run Locally

### 1. Clone repo
```bash
git clone https://github.com/your-username/Personal-assistant.git
cd Personal-assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set environment variables
```bash
export GOOGLE_API_KEY=your_api_key
```

### 4. Run server
```bash
uvicorn main:app --reload
```

### 5. Open in browser
```
http://localhost:8000
```

---

## 🚀 Deployment (Railway)

1. Push code to GitHub  
2. Create project on Railway  
3. Add environment variables:
   ```
   GOOGLE_API_KEY=your_key
   ```
4. Deploy 🚀  

---

## ⚠️ Limitations

- ChromaDB stored locally (resets on redeploy if not persisted)  
- Rate limited to **3 requests/min**  
- Single-user context  

---

## 🔥 Future Improvements

- Streaming responses  
- Persistent vector DB  
- Multi-user support  
- Better query routing  
- Analytics  

---

## 💡 Motivation

This project explores how to build **personalized AI systems** that go beyond generic chatbots and act as **context-aware digital representations of individuals**.

---

## 👤 Author

**Pratyush Khengle**  
Agentic AI / Applied AI Engineer  