import os
import glob
import shutil
from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredImageLoader
)
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


LOADER_MAPPING = {
    ".txt": (TextLoader, {"encoding": "utf-8"}),
    ".md": (TextLoader, {"encoding": "utf-8"}),
    ".py": (TextLoader, {"encoding": "utf-8"}),
    ".json": (TextLoader, {"encoding": "utf-8"}),
    ".csv": (TextLoader, {"encoding": "utf-8"}),
    ".sql": (TextLoader, {"encoding": "utf-8"}),
    ".pdf": (PyPDFLoader, {}),
    ".docx": (Docx2txtLoader, {}),
    ".doc": (Docx2txtLoader, {}),
    ".xlsx": (UnstructuredExcelLoader, {}),
    ".xls": (UnstructuredExcelLoader, {}),
    ".jpg": (UnstructuredImageLoader, {}),
    ".jpeg": (UnstructuredImageLoader, {}),
    ".png": (UnstructuredImageLoader, {})
}

def get_loader_for_path(file_path:str):
    ext = Path(file_path).suffix.lower()

    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        return loader_class(str(file_path), **loader_args)
    return None


def fetch_documents(root: str):
    documents = []
    
    for path in Path(root).rglob("*"):
        
        if path.is_file():
            try:
                loader = get_loader_for_path(str(path))
                doc_list = loader.load()
                
                for doc in doc_list:
                    doc.metadata["doc_type"] = str(path).lower() 
                    documents.append(doc)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                
    return documents

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_embeddings(chunks):
    DB_NAME = "local_vector_store"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    )

    return vectorstore
    
def vectorize_db(db_path: str):
    yield f"Vectorizing:'{db_path}'... (this can take a while)"
    docs = fetch_documents(db_path)
    chunks = create_chunks(docs)
    create_embeddings(chunks)
    yield f"Done! Vector DB is created/updated."