from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline
from src.config import LLM_MODEL_NAME, LLM_PROVIDER, LOCAL_LLM_MODEL_NAME, OPENAI_API_KEY, TOP_K_RESULTS
from src.vector_store import StudyVectorStore

NOT_FOUND_RESPONSE = "I couldn't find the answer to this question in your uploaded study material."

QA_PROMPT_TEMPLATE = "Answer the question using the context. Context: {context} Question: {question}"

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

        if LLM_PROVIDER == "openai":
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=0.3,
                api_key=OPENAI_API_KEY if OPENAI_API_KEY else None,
            )
        elif LLM_PROVIDER in ["local", "huggingface"]:
            try:
                self.llm = HuggingFacePipeline.from_model_id(
                    model_id=LOCAL_LLM_MODEL_NAME,
                    task="text2text-generation",
                    pipeline_kwargs={"max_new_tokens": 512, "truncation": True},
                )
            except Exception:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
                model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_LLM_MODEL_NAME)
                tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL_NAME)
                pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, truncation=True)
                self.llm = HuggingFacePipeline(pipeline=pipe)
        else:
            raise ValueError(f"Unsupported LLM provider: '{LLM_PROVIDER}'")

    def _format_docs(self, docs) -> str:
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def _clean_llm_response(self, raw_response: str) -> str:
        """Strips system prompts, context regurgitation, and internal template text from LLM output."""
        if not raw_response:
            return ""

        text = raw_response

        # If the output repeats "Answer:", extract text after the last "Answer:"
        if "Answer:" in text:
            text = text.split("Answer:")[-1]

        # Remove common prompt headers if present
        for header in [
            "Human:",
            "Assistant:",
            "You are StudyMate AI",
            "Retrieved Context:",
            "Question:",
            "Study Material:",
        ]:
            if header in text:
                text = text.replace(header, "")

        return text.strip()

    def ask_question(self, question: str, top_k: int = TOP_K_RESULTS) -> Dict[str, Any]:
        """Answers a student's question using RAG context."""
        # Retrieve documents with relevance scores
        results = self.vector_store.similarity_search(question, k=top_k)
        docs = [doc for doc, _score in results] if results else []

        if not docs:
            return {
                "question": question,
                "answer": NOT_FOUND_RESPONSE,
                "source_documents": docs,
            }

        context = self._format_docs(docs)
        prompt = ChatPromptTemplate.from_template(QA_PROMPT_TEMPLATE)
        chain = prompt | self.llm | StrOutputParser()
        raw_response = chain.invoke({"context": context, "question": question})

        cleaned_answer = self._clean_llm_response(raw_response)

        return {
            "question": question,
            "answer": cleaned_answer,
            "source_documents": docs,
        }

    def generate_summary(self, top_k: int = 6) -> str:
        """Generates a structured study summary from the vector store context."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke("key concepts overview summary main topics")
        context = self._format_docs(docs)

        prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT_TEMPLATE)
        chain = prompt | self.llm | StrOutputParser()
        raw_res = chain.invoke({"context": context})
        return self._clean_llm_response(raw_res)

    def generate_quiz(self, topic: str = "general", top_k: int = 5) -> str:
        """Generates practice quiz questions based on the document context."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(topic)
        context = self._format_docs(docs)

        prompt = ChatPromptTemplate.from_template(QUIZ_PROMPT_TEMPLATE)
        chain = prompt | self.llm | StrOutputParser()
        raw_res = chain.invoke({"context": context})
        return self._clean_llm_response(raw_res)
