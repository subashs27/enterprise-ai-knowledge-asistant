# 🤖 Enterprise AI Knowledge Assistant

An AI-powered enterprise knowledge assistant that enables employees to query company documents using natural language. Built with Retrieval-Augmented Generation (RAG), the application combines hybrid search, Cross-Encoder reranking, and Google Gemini 2.5 Flash to generate context-aware answers with supporting evidence.

## 🚀 Live Demo

Try the application: [Open Enterprise AI Knowledge Assistant](https://enterprise-ai-knowledge-asistant-fzntfs66grmusbfe5smatt.streamlit.app/)

Note: The application may take a short time to start after a period of inactivity.

## Features

- Hybrid Retrieval: Combines FAISS semantic search and BM25 keyword search.
- Cross-Encoder Reranking: Reranks retrieved content based on relevance.
- Query Rewriting: Refines user queries to improve retrieval.
- Context-Aware Answers: Uses Gemini 2.5 Flash to generate responses from retrieved document content.
- Evidence Viewer: Displays supporting evidence for generated answers.
- Confidence Score: Provides a confidence indicator for responses.
- Chat Management: Supports new conversations and chat history.
- Hallucination Reduction: Grounds responses in retrieved enterprise documents.

## Tech Stack

- Language: Python
- Frontend: Streamlit
- LLM: Google Gemini 2.5 Flash
- RAG Framework: LangChain
- Vector Search: FAISS
- Keyword Search: BM25
- Reranking: Cross-Encoder
- Embeddings: Google Generative AI Embeddings
- Document Processing: PyPDF

## System Architecture

The application follows a Retrieval-Augmented Generation pipeline:

```text
          User Query
              |
              v
         Query Rewriting
              |
              v
         Hybrid Search
         /           \
      FAISS          BM25
         \           /
              |
              v
      Cross-Encoder Reranking
              |
              v
       Top 3 Relevant Chunks
              |
              v
       Google Gemini 2.5 Flash
              |
              v
       Answer + Supporting Evidence
```

## 💡 Sample Questions

Try asking the assistant questions such as:

- What is the Work From Home policy?
- Who is eligible for remote work?
- How much is the internet reimbursement?
- What are the employee leave policies?
- What benefits are provided to employees?
- What are the core working hours?
- Is manager approval required for remote work?
- What are the performance expectations for remote employees?
- Which documents mention insurance benefits?
- Does the company have a mandatory haircut policy?

## Run Locally
1. Clone the repository:

   ```bash
   git clone https://github.com/subashs27/enterprise-ai-knowledge-asistant.git
   cd enterprise-ai-knowledge-asistant
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure the required Google API key securely using Streamlit Secrets or environment variables.

4. Run the application:

   ```bash
   streamlit run app.py
   ```

Ensure the required FAISS index and other application resources are available before running the app.

## Deployment
The application is deployed on Streamlit Community Cloud.
The app may take a short time to start after a period of inactivity.

## Security Notes
- Store API keys securely using Streamlit Cloud Secrets or environment variables.
- Never commit API keys or credentials to GitHub.
- Use authorized documents when testing or demonstrating the application.
- Avoid exposing confidential enterprise information in logs or responses.

## Future Enhancements
- Multi-user authentication and role-based access control
- Advanced analytics dashboard
- Multilingual document support
- Voice-based question answering

## Author
SUBASH S
GitHub: [subashs27](https://github.com/subashs27)
