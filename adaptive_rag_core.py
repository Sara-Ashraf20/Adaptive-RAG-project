# adaptive_rag_core.py
from classifier import classify_query
from rag_single import single_rag
from rag_multi import multi_hop_rag
from llm import get_llm

llm = get_llm()

def no_retrieval(question):
    """Answer using LLM without retrieval"""
    return llm.invoke(question).content

def adaptive_rag(question):
    """Adaptive RAG logic based on query type"""
    label = classify_query(question)

    if label == "A":
        return no_retrieval(question)
    elif label == "B":
        return single_rag(question)
    else:
        return multi_hop_rag(question)
