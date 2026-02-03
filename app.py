import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chains.retrieval import create_retrieval_chain

VECTORSTORE_PATH = "vectorstore"

st.set_page_config(page_title="Financial Document Assistant")
st.title("📊 Financial Document Assistant (Local LLaMA)")

@st.cache_resource
def load_chain():
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
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are a financial analyst assistant.
        Answer the question using ONLY the provided context.
        If the answer is not in the context, say "I don't know".

        Context:
        {context}

        Question:
        {input}
        """
    )

    document_chain = create_stuff_documents_chain(
        llm,
        prompt,
        output_parser=StrOutputParser()
    )

    return create_retrieval_chain(retriever, document_chain)

chain = load_chain()

query = st.text_input("Ask a question about the documents:")

if query:
    result = chain.invoke({"input": query})
    st.subheader("Answer")
    st.write(result.get("answer", result))