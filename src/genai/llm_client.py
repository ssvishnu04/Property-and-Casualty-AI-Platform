import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from .env")

    return ChatGroq(
        groq_api_key=api_key,
        model_name=model,
        temperature=0.1,
    )


def call_llm(prompt: str) -> str:
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content