# rag_multi.py
from llm import get_llm
from retriever import get_retriever

llm = get_llm()
retriever = get_retriever()

MAX_HOPS = 2

def multi_hop_rag(question):
    """Multi-hop RAG for complex queries"""
    context = ""
    current_query = question

    for hop in range(MAX_HOPS):
        docs = retriever.invoke(current_query)
        new_context = "\n".join([d.page_content for d in docs])
        context += "\n" + new_context

        followup_prompt = f"""
We are solving a complex question.

Original Question:
{question}

Current Context:
{context}

What should we search NEXT to answer the question?
Return only the search query.
"""
        current_query = llm.invoke(followup_prompt).content

    final_prompt = f"""
Use the context to answer.

Context:
{context}

Question:
{question}
"""
    return llm.invoke(final_prompt).content
