import argparse
import sys
from pathlib import Path
from src.config import DOCS_DIR
from src.document_loader import StudyDocumentLoader
from src.vector_store import StudyVectorStore
from src.rag_chain import StudyRAGChain


def build_and_index():
    """CLI action to ingest documents from data/docs and build the FAISS index."""
    print("=== StudyMate AI: Indexing Study Materials ===")
    loader = StudyDocumentLoader()
    docs = loader.load_directory(str(DOCS_DIR))

    if not docs:
        print(f"[-] No documents found in '{DOCS_DIR}'. Add PDF, TXT, or MD files to start.")
        return False

    print(f"[+] Loaded {len(docs)} document chunks.")
    vector_store = StudyVectorStore()
    vector_store.build_index(docs)
    vector_store.save_index()
    print("[+] FAISS vector index built and saved successfully!")
    return True


def interactive_qa():
    """CLI action to launch an interactive terminal Q&A session."""
    print("\n=== StudyMate AI: Interactive Study Assistant ===")
    vector_store = StudyVectorStore()
    try:
        vector_store.load_index()
    except FileNotFoundError:
        print("[-] FAISS index not found. Building index first...")
        success = build_and_index()
        if not success:
            return
        vector_store.load_index()

    rag = StudyRAGChain(vector_store=vector_store)
    print("[+] Ready! Ask any question about your study materials (type 'exit' to quit).\n")

    while True:
        try:
            query = input("Student Q: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye! Happy studying!")
                break

            result = rag.ask_question(query)
            print(f"\nStudyMate AI:\n{result['answer']}\n")
            print("Sources:")
            for i, doc in enumerate(result["source_documents"], 1):
                src = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", None)
                page_str = f" (Page {page+1})" if page is not None else ""
                print(f"  [{i}] {Path(src).name}{page_str}")
            print("-" * 50)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"[-] Error answering query: {e}")


def main():
    parser = argparse.ArgumentParser(description="StudyMate AI CLI Manager")
    parser.add_argument(
        "--index",
        action="store_true",
        help="Rebuild FAISS vector index from data/docs directory",
    )
    args = parser.parse_args()

    if args.index:
        build_and_index()
    else:
        interactive_qa()


if __name__ == "__main__":
    main()
