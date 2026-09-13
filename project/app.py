from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch
import os


if __name__ == "__main__":

    faiss_path = "faiss_store/faiss.index"
    meta_path = "faiss_store/metadata.pkl"

    store = FaissVectorStore("faiss_store")

    # Build FAISS index if it doesn't exist
    if not (
        os.path.exists(faiss_path)
        and os.path.exists(meta_path)
    ):

        print(
            "[INFO] FAISS index not found. Building new index..."
        )

        docs = load_all_documents("data")

        store.build_from_documents(docs)

    else:

        print(
            "[INFO] Existing FAISS index found. Loading..."
        )

        store.load()


    # RAG search
    rag_search = RAGSearch()

    query = "what is the result of 8th semester of Fatima aslam?"

    summary = rag_search.search_and_summarize(
        query,
        top_k=3
    )

    print("\nSummary:")
    print(summary)