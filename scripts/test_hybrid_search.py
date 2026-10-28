from app.rag.hybrid_search import hybrid_search

queries = [
    "What caused INC-1001?",
    "What is PAYMENT_DB_TIMEOUT?",
    "How do I troubleshoot CrashLoopBackOff?",
    "How do I rollback a production deployment?",
]


for query in queries:
    print("\n" + "=" * 100)
    print(f"QUERY: {query}")
    print("=" * 100)

    results = hybrid_search(
        query=query,
        top_k=5,
        candidate_k=10,
    )

    for index, result in enumerate(results, start=1):
        print(f"\n[{index}] {result.title}")
        print(f"Hybrid Score : {result.hybrid_score:.6f}")
        print(f"Vector Score : {result.vector_score}")
        print(f"Keyword Score: {result.keyword_score}")
        print(f"Source       : {result.source}")
        print(f"Content      : {result.content[:200]}...")
