import os
from dotenv import load_dotenv

import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API Key
load_dotenv()

# Page Title
st.set_page_config(
    page_title="Enterprise AI Knowledge Assistant",
    page_icon="🤖"
)

st.title("🤖 Enterprise AI Knowledge Assistant")

st.sidebar.title("TechNova AI Solutions")
st.sidebar.write("Enterprise Knowledge Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Load Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# Load Embeddings
GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# Load FAISS
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# Question Input
question = st.chat_input(
    "Ask a question about company policies"
)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question and question.strip():

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
    You are an Enterprise Knowledge Assistant.

    Answer only using the provided context.

    If information is not available,
    say:
    'Information not found in the documents.'

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    with st.chat_message("assistant"):
        st.markdown(response.content)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.content
        }
    )

    st.subheader("📄 Source")

    source = os.path.basename(
        docs[0].metadata["source"]
    )

    source = source.replace(".pdf", "")
    source = source.replace("_", " ")

    st.write(source)