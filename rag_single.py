# rag_single.py
from llm import get_llm
from retriever import get_retriever

llm = get_llm()
retriever = get_retriever()

def single_rag(question):
    """Single-document RAG retrieval"""
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Answer using the context below.

Context:
{context}

Question:
{question}
"""
    return llm.invoke(prompt).content
