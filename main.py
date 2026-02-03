# main.py
from adaptive_rag_core import adaptive_rag

def main():
    print("\n🔥 Adaptive-RAG with Groq Ready!\n")

    while True:
        question = input("Ask (type 'exit' to quit): ")
        if question.lower() == "exit":
            print("Goodbye! 👋")
            break

        try:
            answer = adaptive_rag(question)
        except Exception as e:
            print(f"\n❌ Error generating answer: {e}")
            continue

        print("\n✅ Answer:\n")
        print(answer)
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()
