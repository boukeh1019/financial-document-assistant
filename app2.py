import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

VECTORSTORE_PATH = "vectorstore"

# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(page_title="Financial Document Assistant")
st.title("📊 Financial Document Assistant (Local LLaMA)")

st.write(
    "This assistant answers questions strictly using the provided "
    "financial and regulatory documents."
)

# ----------------------------
# Load components
# ----------------------------
@st.cache_resource
def load_chain():
    # Embeddings MUST match ingest.py
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOllama(
        model="llama3.1:8b",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are a financial analyst assistant.

        Answer the question using ONLY the context below.
        If the answer is not contained in the context, say:
        "I don't know based on the provided documents."

        Context:
        {context}

        Question:
        {question}
        """
    )

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

chain, retriever = load_chain()

# ----------------------------
# User input
# ----------------------------
query = st.text_input(
    "Ask a question about the documents:",
    placeholder="e.g. What credit risks are discussed?"
)

# ----------------------------
# Run query
# ----------------------------
if query:
    with st.spinner("Searching documents..."):
        answer = chain.invoke(query)
        docs = retriever.invoke(query)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Source Documents"):
        for i, doc in enumerate(docs, start=1):
            st.markdown(f"**Source {i}**")
            st.write(doc.metadata)
            st.write(doc.page_content[:500] + "...")
            st.markdown("---")
