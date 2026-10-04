import os
from pathlib import Path
import streamlit as st
from src.config import DOCS_DIR
from src.document_loader import StudyDocumentLoader
from src.vector_store import StudyVectorStore
from src.rag_chain import StudyRAGChain

# Page Configuration
st.set_page_config(
    page_title="StudyMate AI - RAG & FAISS Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎓 StudyMate AI")
st.subheader("Your Intelligent RAG & FAISS-powered AI Study Assistant")

# Sidebar Configuration & Upload
with st.sidebar:
    st.header("⚙️ Document Management")
    st.markdown("Upload your study notes, textbooks, or reference papers (PDF, TXT, MD).")

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for file in uploaded_files:
            save_path = DOCS_DIR / file.name
            with open(save_path, "wb") as f:
                f.write(file.getvalue())
        st.success(f"Saved {len(uploaded_files)} file(s) to `{DOCS_DIR}`.")

    if st.button("🔨 Build / Rebuild Vector Index", use_container_width=True):
        with st.spinner("Processing documents & building FAISS index..."):
            try:
                loader = StudyDocumentLoader()
                docs = loader.load_directory(str(DOCS_DIR))
                if not docs:
                    st.warning("No study documents found in data/docs. Please upload files first.")
                else:
                    vector_store = StudyVectorStore()
                    vector_store.build_index(docs)
                    vector_store.save_index()
                    st.success(f"Successfully indexed {len(docs)} document chunks!")
            except Exception as e:
                st.error(f"Error building index: {e}")

    st.markdown("---")
    st.markdown("### 📌 About StudyMate AI")
    st.caption("Powered by LangChain, FAISS Vector Search, and Generative AI.")

# Initialize Vector Store & RAG Chain in Session State
@st.cache_resource
def load_rag_pipeline():
    vstore = StudyVectorStore()
    vstore.load_index()
    return StudyRAGChain(vector_store=vstore)

# Tabs Interface
tab1, tab2, tab3 = st.tabs(["💬 Q&A Chat", "📝 Study Summary", "🧩 Practice Quiz"])

with tab1:
    st.markdown("#### Ask Questions About Your Study Materials")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("📚 View Retrieved Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"- **{src['name']}** (Page {src['page']}): *\"{src['excerpt']}...\"*")

    user_query = st.chat_input("Ask a question about your study material...")

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching study notes & generating response..."):
                try:
                    rag = load_rag_pipeline()
                    res = rag.ask_question(user_query)
                    answer = res["answer"]
                    sources = []
                    for doc in res["source_documents"]:
                        sources.append({
                            "name": Path(doc.metadata.get("source", "Doc")).name,
                            "page": doc.metadata.get("page", 0) + 1,
                            "excerpt": doc.page_content[:150].replace("\n", " "),
                        })

                    st.markdown(answer)
                    with st.expander("📚 View Retrieved Sources"):
                        for src in sources:
                            st.markdown(f"- **{src['name']}** (Page {src['page']}): *\"{src['excerpt']}...\"*")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                except Exception as e:
                    st.error(f"Error executing RAG search: {e}. Please build the vector index first.")

with tab2:
    st.markdown("#### Generate Comprehensive Study Summary")
    if st.button("✨ Summarize Key Concepts"):
        with st.spinner("Generating summary from knowledge base..."):
            try:
                rag = load_rag_pipeline()
                summary = rag.generate_summary()
                st.markdown(summary)
            except Exception as e:
                st.error(f"Error generating summary: {e}")

with tab3:
    st.markdown("#### Generate Practice Quiz & Flashcard Questions")
    quiz_topic = st.text_input("Enter Topic Focus (Optional)", placeholder="e.g. Neural Networks, Calculus, Data Structures")
    if st.button("🎯 Generate Quiz"):
        with st.spinner("Creating practice quiz..."):
            try:
                rag = load_rag_pipeline()
                quiz = rag.generate_quiz(topic=quiz_topic if quiz_topic else "general")
                st.markdown(quiz)
            except Exception as e:
                st.error(f"Error generating quiz: {e}")
