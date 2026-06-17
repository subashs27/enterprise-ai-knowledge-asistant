from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS database
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# Test query
query = "How do employees request leave?"

results = retriever.invoke(query)

print("\nTop Retrieved Chunks:\n")

for i, doc in enumerate(results, start=1):
    print(f"\nResult {i}:")
    print(doc.page_content[:500])
    print("-" * 50)