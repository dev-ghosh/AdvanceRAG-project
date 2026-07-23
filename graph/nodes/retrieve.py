from typing import Any, Dict

from ..state import GraphState
from adv_ingestion import retriever

from ..retrieval.reranker import rerank_documents


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)

    documents = rerank_documents(
        question,
        documents,
        top_k=3,
    )

    return {"documents": documents, "question": question}