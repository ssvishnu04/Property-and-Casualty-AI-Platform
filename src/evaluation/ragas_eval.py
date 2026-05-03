import json
from pathlib import Path

import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextPrecision,
    ContextRecall,
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
import os

from src.genai.rag_pipeline import rag_service


load_dotenv()


TEST_DATA_PATH = Path("data/rag_eval/test_questions.json")
REPORT_DIR = Path("reports/ragas")
RESULT_JSON_PATH = REPORT_DIR / "ragas_results.json"
RESULT_CSV_PATH = REPORT_DIR / "ragas_results.csv"


def load_test_questions() -> list[dict]:
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test data not found: {TEST_DATA_PATH}")

    with TEST_DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_ragas_dataset(test_questions: list[dict]) -> Dataset:
    rows = []

    for item in test_questions:
        question = item["question"]
        ground_truth = item["ground_truth"]

        rag_response = rag_service.answer(
            question=question,
            context_data={},
        )

        rows.append(
            {
                "question": question,
                "answer": rag_response["answer"],
                "contexts": rag_response["retrieved_contexts"],
                "ground_truth": ground_truth,
            }
        )

    return Dataset.from_list(rows)


def get_ragas_llm():
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from .env")

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model,
        temperature=0.0,
    )

    return LangchainLLMWrapper(llm)


def get_ragas_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return LangchainEmbeddingsWrapper(embeddings)


def run_ragas_evaluation() -> None:
    print("Starting RAGAS evaluation...")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test questions...")
    test_questions = load_test_questions()

    print(f"Loaded {len(test_questions)} questions")

    print("Building dataset using RAG pipeline...")
    dataset = build_ragas_dataset(test_questions)

    print("Dataset built successfully")

    print("Initializing evaluator LLM...")
    evaluator_llm = get_ragas_llm()

    print("Initializing embeddings...")
    evaluator_embeddings = get_ragas_embeddings()

    print("Running evaluation...")
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    print("Evaluation completed")

    result_df = result.to_pandas()

    result_df.to_csv(RESULT_CSV_PATH, index=False)

    with RESULT_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(result_df.to_dict(orient="records"), file, indent=2)

    print("Saved results")
    print(f"Results saved to {RESULT_JSON_PATH} and {RESULT_CSV_PATH}")
    print(result_df)


if __name__ == "__main__":
    run_ragas_evaluation()