# classifier.py
from llm import get_llm

llm = get_llm()

def classify_query(question):
    prompt = f"""
You are a query complexity classifier.

Return ONLY one letter:

A → Simple question answerable from general knowledge  
B → Needs retrieval from documents  
C → Requires multi-step reasoning across multiple documents  

Question: {question}
"""
    response = llm.invoke(prompt).content.strip()
    label = response[0] if response else "B"

    if label not in ["A", "B", "C"]:
        label = "B"

    print(f"\n[Adaptive-RAG] Query classified as: {label}\n")
    return label
