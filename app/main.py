from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import AnalyzeRequest, AnalyzeResponse, CommentAnalysis
from app.services.azure_client import generate_response, generate_summary
from app.services.classifier import classifier

app = FastAPI(title="Comment Intelligence API", version="0.1.0")

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4173", "http://localhost:5173", "http://127.0.0.1:4173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def summarize_prompt(comment: str, response: str | None) -> str:
    if response:
        return (
            "Summarize the following public comment and reference the agency's prior response if relevant.\n"
            f"Comment: {comment}\n"
            f"Prior Response: {response}"
        )
    return (
        "Summarize the following public comment into 2-3 concise sentences, highlight key issues and requested actions.\n"
        f"Comment: {comment}"
    )


def response_prompt(comment: str, summary: str, label: str, prior_response: str | None) -> str:
    base = [
        "Draft a short agency response (<=150 words) acknowledging the comment, referencing key points,",
        "and clarifying next steps. Maintain neutral, factual tone.",
        f"Summary: {summary}",
        f"Classification: {label}",
        f"Comment: {comment}",
    ]
    if prior_response:
        base.append(f"Prior Response: {prior_response}")
        base.append("Only offer a new response if additional info is needed; otherwise restate key commitments succinctly.")
    return "\n".join(base)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    if not classifier.is_ready():
        raise HTTPException(status_code=503, detail="Classifier model not found. Train notebook and export model first.")

    results: list[CommentAnalysis] = []
    for item in request.items:
        classification = classifier.predict(item.comment)
        if classification is None:
            raise HTTPException(status_code=503, detail="Classifier unavailable.")

        summary = generate_summary(summarize_prompt(item.comment, item.prior_response))
        generated_response = generate_response(
            response_prompt(
                comment=item.comment,
                summary=summary,
                label=classification["label"],
                prior_response=item.prior_response,
            )
        )

        results.append(
            CommentAnalysis(
                comment_id=item.comment_id,
                summary=summary,
                label=classification["label"],
                probability=classification["probability"],
                generated_response=generated_response,
            )
        )

    return AnalyzeResponse(results=results)
