from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load PDFs
loader = PyPDFDirectoryLoader("data")
documents = loader.load()

# Chunk documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS database
vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

# Save locally
vectorstore.save_local("faiss_index")

print("FAISS Vector Database Created Successfully!")
print(f"Total Chunks Stored: {len(chunks)}")