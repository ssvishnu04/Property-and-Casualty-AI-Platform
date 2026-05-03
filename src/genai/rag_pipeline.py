from typing import Any

from src.genai.retriever import load_documents, simple_retriever
from src.genai.prompt_templates import claim_explanation_prompt
from src.genai.llm_client import call_llm


class RAGService:
    def __init__(self):
        self.docs = load_documents()

    def answer(
        self,
        question: str,
        context_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        retrieved_docs = simple_retriever(
            query=question,
            docs=self.docs,
            top_k=2,
            min_score=0.30,
        )

        retrieved_context = "\n\n".join(
            [
                f"Source: {doc['file_path']}\n{doc['content']}"
                for doc in retrieved_docs
            ]
        )

        structured_context = context_data or {}

        combined_context = (
            f"Retrieved Policy/Treaty/Operations Context:\n"
            f"{retrieved_context}\n\n"
            f"Structured ML Prediction Context:\n"
            f"{structured_context}"
        )

        prompt = claim_explanation_prompt(
            context=combined_context,
            question=question,
        )

        answer = call_llm(prompt)

        return {
            "answer": answer,
            "retrieved_contexts": [doc["content"] for doc in retrieved_docs],
            "retrieved_sources": [
                {
                    "file_path": doc["file_path"],
                    "score": float(doc["score"]),
                    "similarity_score": float(doc["similarity_score"]),
                }
                for doc in retrieved_docs
            ],
        }


rag_service = RAGService()