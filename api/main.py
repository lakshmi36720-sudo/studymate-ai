from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.config import DOCS_DIR
from src.document_loader import StudyDocumentLoader
from src.rag_chain import StudyRAGChain
from src.vector_store import StudyVectorStore

app = FastAPI(title="StudyMate AI API")
UPLOAD_DIR = DOCS_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1)


class SummaryRequest(BaseModel):
    top_k: int = Field(default=6, ge=1)


class QuizRequest(BaseModel):
    topic: str = Field(default="general", min_length=1)
    top_k: int = Field(default=5, ge=1)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "StudyMate AI API"}


def _get_vector_store() -> StudyVectorStore:
    vector_store = StudyVectorStore()
    try:
        vector_store.load_index()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail="No indexed documents found. Upload files first.") from exc
    return vector_store


@app.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)) -> dict[str, object]:
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    uploaded_files: list[Path] = []
    loader = StudyDocumentLoader()
    documents = []

    for upload in files:
        if not upload.filename:
            raise HTTPException(status_code=400, detail="Each uploaded file must have a filename.")

        file_path = UPLOAD_DIR / f"{uuid4()}_{Path(upload.filename).name}"
        with file_path.open("wb") as destination:
            while chunk := await upload.read(1024 * 1024):
                destination.write(chunk)

        uploaded_files.append(file_path)

        try:
            documents.extend(loader.load_single_document(str(file_path)))
        except ValueError as exc:
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {upload.filename}") from exc
        except Exception as exc:
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"Could not process {upload.filename}: {exc}") from exc

    if not documents:
        for file_path in uploaded_files:
            file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="No readable content was found in the uploaded files.")

    try:
        vector_store = StudyVectorStore()
        vector_store.build_index(documents)
        vector_store.save_index()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not index uploaded documents: {exc}") from exc

    return {
        "message": f"Indexed {len(documents)} document chunks from {len(uploaded_files)} file(s).",
        "files": [path.name for path in uploaded_files],
    }


@app.post("/ask")
def ask_question(request: AskRequest) -> dict[str, object]:
    vector_store = _get_vector_store()
    rag = StudyRAGChain(vector_store=vector_store)
    return rag.ask_question(request.question, top_k=request.top_k)


@app.post("/summary")
def generate_summary(request: SummaryRequest) -> dict[str, str]:
    vector_store = _get_vector_store()
    rag = StudyRAGChain(vector_store=vector_store)
    return {"summary": rag.generate_summary(top_k=request.top_k)}


@app.post("/quiz")
def generate_quiz(request: QuizRequest) -> dict[str, str]:
    vector_store = _get_vector_store()
    rag = StudyRAGChain(vector_store=vector_store)
    return {"quiz": rag.generate_quiz(topic=request.topic, top_k=request.top_k)}
