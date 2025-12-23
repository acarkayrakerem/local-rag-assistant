from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from model_config import *


DB_NAME = str(Path(__file__).parent / "local_vector_store")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


def fetch_context(question: str) -> list[Document]:
    vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    return retriever.invoke(question)


def combined_question(question: str, history: list[dict] = []) -> str:
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


def answer_question(question: str, llm, history: list[dict] = [],reranker_feature: bool = False) -> str:
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    if reranker_feature:
        docs = rerank(question, docs, llm)
    context = "\n\n".join(f"Source: {doc.metadata.get('doc_type', 'Unknown')}\n\n{doc.page_content}" for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content

def rerank(question, chunks, llm):
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    structured_llm = llm.with_structured_output(RankOrder)
    response = structured_llm.invoke(messages)
    order = response.order
    return [chunks[i - 1] for i in order]