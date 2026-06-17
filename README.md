# Enterprise AI Knowledge Assistant

## Overview

Enterprise AI Knowledge Assistant is a Retrieval-Augmented Generation (RAG) application that helps employees get instant answers from company policy documents.

The system uses LangChain, FAISS, HuggingFace Embeddings, Google Gemini 2.5 Flash, and Streamlit to retrieve relevant information and generate accurate responses with source citations.

---

## Features

- Natural language question answering
- Company policy document search
- FAISS vector database
- Gemini 2.5 Flash integration
- Source citation support
- Streamlit chat interface

---

## Tech Stack

- Python
- LangChain
- FAISS
- HuggingFace Embeddings
- Google Gemini 2.5 Flash
- Streamlit

---

## Architecture

Company PDF Files

↓

PDF Loader

↓

Text Chunking

↓

Embeddings

↓

FAISS Vector Store

↓

Retriever

↓

Gemini 2.5 Flash

↓

Streamlit Chat UI

↓

Answer + Source Citation

---

## Sample Questions

- How many annual leave days are employees entitled to?
- How do employees submit leave requests?
- What is the company dress code?
- What are the performance review guidelines?

---

## Future Enhancements

- Multi-document source citations
- PDF source links
- Conversation memory
- Role-based document access
