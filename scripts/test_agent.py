import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.graph.graph import graph


def main() -> None:
    question = "What causes CrashLoopBackOff in Kubernetes?"

    result = graph.invoke(
        {
            "question": question,
        }
    )

    print("\n" + "=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(question)

    print("\n" + "=" * 70)
    print("MESSAGE FLOW")
    print("=" * 70)

    for index, message in enumerate(
        result.get("messages", []),
        start=1,
    ):
        print(f"\n--- Message {index} ---")

        print(f"Type: {type(message).__name__}")

        if getattr(message, "content", None):
            print(f"Content:\n{message.content}")

        if getattr(message, "tool_calls", None):
            print(f"Tool Calls:\n{message.tool_calls}")

        if getattr(message, "name", None):
            print(f"Tool Name: {message.name}")

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    messages = result.get("messages", [])

    if messages:
        print(messages[-1].content)


if __name__ == "__main__":
    main()
