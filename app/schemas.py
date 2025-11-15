from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class CommentPayload(BaseModel):
    comment_id: Optional[str] = Field(default=None)
    comment: str
    prior_response: Optional[str] = None


class AnalyzeRequest(BaseModel):
    items: List[CommentPayload]


class CommentAnalysis(BaseModel):
    comment_id: Optional[str] = None
    summary: str
    label: str
    probability: float
    generated_response: str


class AnalyzeResponse(BaseModel):
    results: List[CommentAnalysis]
