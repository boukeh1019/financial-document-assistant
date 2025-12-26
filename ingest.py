import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/documents"
VECTORSTORE_PATH = "vectorstore"

def ingest_documents():
    documents = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH,file))
            documents.extend(loader.load())


    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(chunks,embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)


    print(f"Ingested {len(chunks)} chunks.")


if __name__ == "__main__":
    ingest_documents()