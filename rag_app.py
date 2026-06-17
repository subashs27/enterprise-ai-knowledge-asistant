import os
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Gemini Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS Database
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# User Question
question = input("Ask a Question: ")

# Retrieve Relevant Chunks
docs = retriever.invoke(question)
sources = set()

for doc in docs:
    source = doc.metadata.get("source", "Unknown")
    sources.add(source)

context = "\n\n".join(
    [doc.page_content for doc in docs]
)

# Prompt
prompt = f"""
You are an Enterprise Knowledge Assistant.

Answer only using the provided context.

If the answer is not available in the context,
reply with:
'Information not found in the documents.'

Context:
{context}

Question:
{question}
"""

response = llm.invoke(prompt)

print("\nAnswer:\n")
print(response.content)

top_source = docs[0].metadata["source"]

print("\nSource:")
print("-", os.path.basename(top_source))