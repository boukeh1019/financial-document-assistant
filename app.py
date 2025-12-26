import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.chains import RetrievalQA


VECTORSTORE_PATH = "vectorstore"

st.set_page_config(page_title="Financial Document Assistant")
st.title("Financial Document Assistant (Local LLaMA)")

@st.cache_resource
def load_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-tranformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.load_local(VECTORSTORE_PATH,embeddings,allow_dangerous_deserialization-True)

    llm = Ollama(model="llama3.1:8b",temperature=0)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k":4}),
        return_source_documents=True       

    )

qa_chain = load_qa_chain()

query = st.text_input("Ask a question about the documents:")

if query:
    result = qa_chain(query)
    st.subheader("Answer")
    st.write(result["result"])

    with st.expander("Sources"):
        for doc in result["source_documents"]:
            st.write(doc.metadata)