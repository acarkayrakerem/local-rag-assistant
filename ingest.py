import os
import glob
import shutil
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic


class ModelConfig:
    def __init__(self, provider, api_key=None, model_name=None):
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name

def get_llm(config: ModelConfig):
    if(config.provider == "openai"):
        return create_openai_llm(config)
    elif(config.provider == "google"):
        return create_google_llm(config)
    elif(config.provider == "anthropic"):
        return create_anthropic_llm(config)
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")

def create_openai_llm(config: ModelConfig):
    return ChatOpenAI(
        api_key=config.api_key,
        model=config.model_name or "gpt-4.1-mini",
        temperature=0.0,
    )
    
def create_google_llm(config: ModelConfig):
    os.environ["GOOGLE_API_KEY"] = config.api_key
    return ChatGoogleGenerativeAI(
        model=config.model_name or "gemini-2.5-flash",
        temperature=0.0,  
    )


def create_anthropic_llm(config: ModelConfig):
    os.environ["ANTHROPIC_API_KEY"] = config.api_key
    return ChatAnthropic(
        model=config.model_name or "claude-haiku-4-5-20251001",
        temperature=0.0,
    )


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
                    doc.metadata["doc_type"] = path.lower() 
                    documents.append(doc)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                
    return documents

def create_chunks(documents, chunk_size=500, chunk_overlap=200 ):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)
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

    vectorstore.persist()
    return vectorstore

