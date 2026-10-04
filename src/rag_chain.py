from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from src.config import LLM_MODEL_NAME, OPENAI_API_KEY, TOP_K_RESULTS
from src.vector_store import StudyVectorStore


QA_PROMPT_TEMPLATE = """You are StudyMate AI, an expert AI study assistant.
Use the following retrieved study context to answer the student's question accurately, concisely, and clearly.
If the answer cannot be found in the provided context, state that clearly instead of making things up.

Retrieved Context:
{context}

Question:
{question}

Answer:"""

SUMMARY_PROMPT_TEMPLATE = """You are StudyMate AI.
Summarize the following study material into key concepts, bullet points, and actionable study takeaways.

Study Material:
{context}

Summary & Key Takeaways:"""

QUIZ_PROMPT_TEMPLATE = """You are StudyMate AI.
Generate 5 practice quiz questions (with detailed answer keys and explanations) based ONLY on the provided study material.

Study Material:
{context}

Practice Quiz:"""


class StudyRAGChain:
    """RAG pipeline coordinator using FAISS retrieval and LLM completion."""

    def __init__(self, vector_store: StudyVectorStore, model_name: str = LLM_MODEL_NAME):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.3,
            api_key=OPENAI_API_KEY if OPENAI_API_KEY else None,
        )

    def _format_docs(self, docs) -> str:
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def ask_question(self, question: str, top_k: int = TOP_K_RESULTS) -> Dict[str, Any]:
        """Answers a student's question using RAG context."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(question)
        context = self._format_docs(docs)

        prompt = ChatPromptTemplate.from_template(QA_PROMPT_TEMPLATE)
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": context, "question": question})

        return {
            "question": question,
            "answer": response,
            "source_documents": docs,
        }

    def generate_summary(self, top_k: int = 6) -> str:
        """Generates a structured study summary from the vector store context."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke("key concepts overview summary main topics")
        context = self._format_docs(docs)

        prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT_TEMPLATE)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context})

    def generate_quiz(self, topic: str = "general", top_k: int = 5) -> str:
        """Generates practice quiz questions based on the document context."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(topic)
        context = self._format_docs(docs)

        prompt = ChatPromptTemplate.from_template(QUIZ_PROMPT_TEMPLATE)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context})
