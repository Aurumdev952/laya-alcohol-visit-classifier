"""HTTP interface for local Laya sentiment and decision recipes."""

from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator

from sentiment_api.service import (
    GpuUnavailableError,
    MODEL_ID,
    SentimentService,
    TextTooLongError,
    load_service,
)


class SentimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str

    @field_validator("text")
    @classmethod
    def text_must_contain_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty or whitespace only")
        return value


class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]
    probabilities: dict[str, float]


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class RecipeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SupportTicketRequest(RecipeRequest):
    body: NonEmptyText


class ModelRoutingRequest(RecipeRequest):
    request: NonEmptyText


class GuardrailsRequest(RecipeRequest):
    prompt: NonEmptyText


class RagFilteringRequest(RecipeRequest):
    question: NonEmptyText
    passage: NonEmptyText


class ModerationRequest(RecipeRequest):
    message: NonEmptyText


class PhishingRequest(RecipeRequest):
    subject: NonEmptyText
    body: NonEmptyText


class RecipeResponse(BaseModel):
    answers: dict[str, dict[str, Any]]


def create_app(service: SentimentService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service if service is not None else load_service()
        yield

    app = FastAPI(title="Laya Decision API", version="0.2.0", lifespan=lifespan)

    def run_recipe(name: str, request: RecipeRequest):
        active: SentimentService = app.state.service
        try:
            return active.predict_recipe(name, request.model_dump())
        except TextTooLongError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GpuUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/health")
    def health():
        active: SentimentService = app.state.service
        if not active.is_ready():
            raise HTTPException(status_code=503, detail="CUDA model is unavailable")
        return {"status": "ok", "device": "cuda", "gpu": active.gpu_name, "model": MODEL_ID}

    @app.post("/sentiment", response_model=SentimentResponse)
    def sentiment(request: SentimentRequest):
        active: SentimentService = app.state.service
        try:
            return active.predict(request.text)
        except TextTooLongError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GpuUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/recipes/support-ticket-triage", response_model=RecipeResponse)
    def support_ticket_triage(request: SupportTicketRequest):
        return run_recipe("support-ticket-triage", request)

    @app.post("/recipes/model-routing", response_model=RecipeResponse)
    def model_routing(request: ModelRoutingRequest):
        return run_recipe("model-routing", request)

    @app.post("/recipes/llm-guardrails", response_model=RecipeResponse)
    def llm_guardrails(request: GuardrailsRequest):
        return run_recipe("llm-guardrails", request)

    @app.post("/recipes/rag-filtering", response_model=RecipeResponse)
    def rag_filtering(request: RagFilteringRequest):
        return run_recipe("rag-filtering", request)

    @app.post("/recipes/content-moderation", response_model=RecipeResponse)
    def content_moderation(request: ModerationRequest):
        return run_recipe("content-moderation", request)

    @app.post("/recipes/phishing-detection", response_model=RecipeResponse)
    def phishing_detection(request: PhishingRequest):
        return run_recipe("phishing-detection", request)

    return app


app = create_app()
