from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)

def rerank_documents(question, documents, top_k=3):

    pairs = [
        (question, doc.page_content)
        for doc in documents
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True,
    )

    return [
        doc
        for _, doc in ranked[:top_k]
    ]