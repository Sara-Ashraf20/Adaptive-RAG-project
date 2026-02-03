# llm.py
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, MODEL_NAME

def get_llm(temp=0):
    """Returns a ChatGroq instance for LLM calls"""
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=temp
    )
