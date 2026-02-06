import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ======================================
# STREAMLIT UI
# ======================================
st.set_page_config(page_title="Strict RAG Chatbot", layout="centered")
st.title(" Strict RAG Chatbot with History")

if "rag_chain" not in st.session_state:

    # Step 1: Load PDF
    loader = PyPDFLoader("attention.pdf")
    documents = loader.load()

    # Step 2: Split
    splitter = RecursiveCharacterTextSplitter()
    docs = splitter.split_documents(documents)

    # Step 3: Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Step 4: Vector Store
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    # Step 5: LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Step 6: Strict Prompt
    prompt = PromptTemplate(
        input_variables=["context", "question", "history"],
        template="""
You are a strict retrieval QA assistant.
Use only the following context and chat history to answer the question.
If the answer is not present, reply exactly with:
"I don't know, the document doesnot contain this information."

Context:
{context}

Chat history:
{history}

Question:
{question}

Answer:
"""
    )

    # Step 7: RetrievalQA Chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    # Step 8: Chat History Store
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    rag_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="query",
        history_messages_key="history"
    )

    st.session_state.rag_chain = rag_with_history


SESSION_ID = "streamlit-session-1"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
query = st.chat_input("Ask a question from the PDF...")

if query:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )
    with st.chat_message("user"):
        st.write(query)

    # Invoke RAG
    response = st.session_state.rag_chain.invoke(
        {"query": query},
        config={"configurable": {"session_id": SESSION_ID}}
    )

    answer = response["result"]

    # Show assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    with st.chat_message("assistant"):
        st.write(answer)
