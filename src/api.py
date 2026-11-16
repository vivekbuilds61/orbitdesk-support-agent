
"""FastAPI interface for the OrbitDesk Support Agent."""

import logging
import time
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from .graph import run_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OrbitDesk Support Agent API",
    description="API for the OrbitDesk LangGraph and RAG support agent.",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


class Citation(BaseModel):
    source_id: str
    passage: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    classification: Literal[
        "answerable",
        "requires_clarification",
        "requires_escalation",
        "out_of_scope",
        "safe_failure",
    ]
    requires_escalation: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    clarification_question: str | None = None
    warnings: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
    elapsed_seconds: float


@app.get("/")
def root():
    return {
        "name": "OrbitDesk Support Agent API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    question = payload.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question must not be blank.",
        )

    started = time.perf_counter()

    try:
        # Run the existing synchronous LangGraph workflow without
        # blocking FastAPI's asynchronous event loop.
        state = await run_in_threadpool(run_graph, question)
    except Exception as exc:
        logger.exception("OrbitDesk agent execution failed")
        raise HTTPException(
            status_code=500,
            detail="The support agent could not process this question.",
        ) from exc

    elapsed = round(time.perf_counter() - started, 3)

    try:
        return ChatResponse(
            answer=state.get(
                "answer",
                "I could not produce a verified answer.",
            ),
            citations=state.get("sources", []),
            classification=state.get(
                "classification", "safe_failure"
            ),
            requires_escalation=bool(
                state.get("requires_human", False)
            ),
            confidence=float(state.get("confidence", 0.0)),
            reason=state.get("reason", ""),
            clarification_question=state.get(
                "clarification_question"
            ),
            warnings=state.get("warnings", []),
            trace=state.get("trace", []),
            elapsed_seconds=elapsed,
        )
    except Exception as exc:
        logger.exception("Invalid final agent state")
        raise HTTPException(
            status_code=500,
            detail="The agent returned an invalid response.",
        ) from exc
