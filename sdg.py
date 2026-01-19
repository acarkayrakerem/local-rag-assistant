import random
from pathlib import Path
from typing import List
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from model_config import *

DB_NAME = str(Path(__file__).parent / "local_vector_store")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

SYSTEM_PROMPT = """
You are an expert data annotator. 
Task: Generate 10 complex questions that can only be answered using the provided context.
Also provide the ground_truth answer. Format: JSON {{question: '', answer: ''}}
"""

USER_PROMPT = """
Context: {context} \n\n\n
Generate the question answer pairs in JSON Format: {{question: '', answer: ''}}
"""

#Pydantic Schema
class QAPair(BaseModel):
    """A single question and its corresponding answer based on context."""
    question: str = Field(description="The complex question generated from the context.")
    answer: str = Field(description="The ground truth answer found in the context.")

class SyntheticDataset(BaseModel):
    """
    A collection of 10 highly challenging, multi-step synthetic 
    questions and answers designed to stress-test a RAG system.
    """
    pairs: List[QAPair]


def fetch_random_context(n: int = 10) -> list:

    if not Path(DB_NAME).exists():
        raise ValueError("Vector store not found. Please click the 'Vectorize Database' button first.")
    vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
    collection = vectorstore._collection
    all_ids = collection.get()['ids']
    
    if len(all_ids) < n:
        n = len(all_ids)
    random_ids = random.sample(all_ids, n)
    raw =  vectorstore.get(ids=random_ids)
    documents = raw.get("documents", [])
    metadatas = raw.get("metadatas", [])

    return [
        Document(page_content=text, metadata=(meta or {}))
        for text, meta in zip(documents, metadatas)
    ]

def generate_sd(llm):

    structured_llm = llm.with_structured_output(SyntheticDataset)
    
    docs = fetch_random_context()
    context = "\n\n".join(f"Source: {doc.metadata.get('doc_type', 'Unknown')}\n\n{doc.page_content}" for doc in docs)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT.format(context=context))
    ]

    dataset = structured_llm.invoke(messages)  

    return dataset

    