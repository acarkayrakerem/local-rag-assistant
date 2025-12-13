import os
import glob
import shutil
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


SUPPORTED_TEXT_EXTENSIONS = {
    ".txt", ".md", ".mdx", ".py", ".json", ".csv", ".tsv", ".log", ".html", ".htm"
}

def fetch_documents(root: str):
    documents = []
    
    for path in Path(root).rglob("*"):

        if path.is_file() and path.suffix.lower() in SUPPORTED_TEXT_EXTENSIONS:
            
            try:
                loader = TextLoader(str(path), encoding="utf-8")
                doc_list = loader.load()
                
                for doc in doc_list:
                    doc.metadata["doc_type"] = str(path).lower() 
                    documents.append(doc)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                
    return documents

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_embeddings(chunks):
    DB_NAME = "local_vector_store"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(DB_NAME):
        shutil.rmtree(DB_NAME)

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    )

    return vectorstore
    

if __name__ == "__main__":
    docs = fetch_documents("test_DB")
    chunks = create_chunks(docs)
    create_embeddings(chunks)