import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.graph.tools import search_knowledge_base
from app.rag.llm import get_llm


def main() -> None:
    llm = get_llm()

    llm_with_tools = llm.bind_tools([search_knowledge_base])

    response = llm_with_tools.invoke(
        [
            (
                "system",
                """
You are a DevOps AI assistant.

You have access to a tool called search_knowledge_base.

For questions about Kubernetes, incidents, deployments,
runbooks, services, or enterprise DevOps documentation,
you MUST call search_knowledge_base before answering.

Do not answer from your own knowledge when the tool can
provide relevant enterprise information.
""",
            ),
            (
                "human",
                "What causes CrashLoopBackOff in Kubernetes?",
            ),
        ]
    )

    print("\n" + "=" * 70)
    print("RESPONSE TYPE")
    print("=" * 70)
    print(type(response).__name__)

    print("\n" + "=" * 70)
    print("CONTENT")
    print("=" * 70)
    print(response.content)

    print("\n" + "=" * 70)
    print("TOOL CALLS")
    print("=" * 70)
    print(response.tool_calls)

    print("\n" + "=" * 70)
    print("ADDITIONAL KWARGS")
    print("=" * 70)
    print(response.additional_kwargs)


if __name__ == "__main__":
    main()
