from langchain_community.document_loaders import PyPDFDirectoryLoader

loader = PyPDFDirectoryLoader("data")
documents = loader.load()

print(f"Total Pages Loaded: {len(documents)}")
print("\nFirst 500 Characters:\n")
print(documents[0].page_content[:500])