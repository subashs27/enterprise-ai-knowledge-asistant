# Enterprise AI Knowledge Assistant

## 📖 Overview

Enterprise AI Knowledge Assistant is a Retrieval-Augmented Generation (RAG) application that enables employees to query company documents using natural language. The system combines Hybrid Search (FAISS + BM25), Cross-Encoder Re-ranking, and Google's Gemini 2.5 Flash to generate accurate, context-aware answers with supporting evidence.

The application minimizes hallucinations by restricting responses to retrieved document content and displays evidence along with confidence scores for transparency.

---

## Features

Hybrid Retrieval (FAISS + BM25)
Cross-Encoder Re-ranking
Query Rewriting
Enterprise RAG Pipeline
Gemini 2.5 Flash
Evidence Viewer
Confidence Score
New Chat
Chat History
Hallucination Prevention
Streamlit UI

---

## Tech Stack

Python
Streamlit
LangChain
FAISS
BM25
Sentence Transformers
Gemini API

---

## Architecture

User Query
      │
      ▼
Query Rewrite
      │
      ▼
Hybrid Search
 (FAISS + BM25)
      │
      ▼
Cross Encoder
      │
      ▼
Top 3 Chunks
      │
      ▼
Gemini 2.5 Flash
      │
      ▼
Answer + Evidence

## Sample Questions

Try asking questions like:

- What is the Work From Home policy?
- Who is eligible for remote work?
- How much is the internet reimbursement?
- What are the employee leave policies?
- What benefits are provided to employees?
- What are the core working hours?
- Is manager approval required for remote work?
- What are the performance expectations for remote employees?
- What documents mention insurance benefits?
- Does the company have a mandatory haircut policy?

---

## Future Enhancements

- Multi-user authentication and role-based access
- Advanced analytics dashboard
- Multi-language document support
- Voice-based question answering