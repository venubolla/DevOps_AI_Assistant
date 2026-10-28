from app.rag.keyword_search import keyword_search

queries = [
    "INC-1001",
    "PAYMENT_DB_TIMEOUT",
    "payment-service v2.4.1",
    "CrashLoopBackOff",
]


for query in queries:
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    results = keyword_search(
        query=query,
        top_k=5,
    )

    for index, result in enumerate(results, start=1):
        print(f"\n[{index}]")
        print(f"Title: {result.title}")
        print(f"Score: {result.score:.4f}")
        print(f"Source: {result.source}")
        print(f"Content: {result.content[:250]}...")
