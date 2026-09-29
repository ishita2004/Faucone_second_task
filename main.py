import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rag_chain import RAGChain

def run_cli():
    print("=" * 65)
    print(" 🤖 Document Q&A Assistant (RAG Pipeline CLI)")
    print("=" * 65)
    print("Initializing RAG Engine (Loading Vector Store & Embeddings)...")
    
    try:
        rag = RAGChain()
        print("Ready! Type your question below (or type 'exit' / 'q' to quit).\n")
    except Exception as e:
        print(f"Error loading vector store: {e}")
        print("Please run ingestion first using: python -m src.ingest")
        return

    while True:
        try:
            user_input = input("Question: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "q", "quit"]:
                print("Goodbye!")
                break

            result = rag.query(user_input)
            print("\n" + "-" * 50)
            print("💡 ANSWER:")
            print(result["answer"])
            print("\n📚 SOURCE CITATIONS:")
            for idx, src in enumerate(result["sources"], start=1):
                print(f"  [{idx}] {src['source']} (Page {src['page']}/{src['total_pages']}) - Score: {src['similarity_score']:.4f}")
            print("-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error executing query: {e}\n")

if __name__ == "__main__":
    run_cli()
