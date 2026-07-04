"""Pydantic request/response schemas for API validation.

Gradual migration — existing routes can adopt these models incrementally
by replacing `body = await request.json()` with Pydantic-validated models.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ── Chat ─────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Validated chat/satisfaction message."""
    task_id: str = Field(..., min_length=1, max_length=64, description="Task identifier")
    message: str = Field(default="", max_length=10000, description="User message text")
    history: list[dict[str, Any]] = Field(default_factory=list, description="Chat history")
    action: str | None = Field(default=None, max_length=64, description="Chat action type")

    @field_validator("task_id")
    @classmethod
    def task_id_alphanumeric(cls, v: str) -> str:
        import re
        if not re.fullmatch(r"[0-9A-Za-z_-]+", v):
            raise ValueError(f"task_id contains invalid characters: {v}")
        return v


class ChatHistorySaveRequest(BaseModel):
    """Save chat history to backend."""
    task_id: str = Field(..., min_length=1, max_length=64)
    history: list[dict[str, Any]] = Field(default_factory=list)


# ── Satisfaction ─────────────────────────────────────────────────────────

class SatisfactionRequest(BaseModel):
    """User satisfaction feedback on outline or content."""
    task_id: str = Field(..., min_length=1, max_length=64)
    stage: str = Field(..., description="'outline' or 'content'")
    satisfied: bool = Field(..., description="Whether user is satisfied")
    feedback: str = Field(default="", max_length=5000, description="Feedback text if dissatisfied")
    version: int = Field(default=0, ge=0, description="Version number being rated")


# ── Generation ───────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    """Start document generation."""
    task_id: str = Field(..., min_length=1, max_length=64)
    prompt: str = Field(default="", max_length=20000, description="Generation prompt")


# ── Version Management ───────────────────────────────────────────────────

class VersionSelectRequest(BaseModel):
    """Select a specific version as the active one."""
    task_id: str = Field(..., min_length=1, max_length=64)
    type: str = Field(..., pattern=r"^(outline|content)$")
    version: int = Field(..., ge=1)


class VersionDeleteRequest(BaseModel):
    """Delete a version snapshot."""
    task_id: str = Field(..., min_length=1, max_length=64)
    type: str = Field(..., pattern=r"^(outline|content)$")
    version: int = Field(..., ge=1)


class VersionRedoRequest(BaseModel):
    """Redo generation from a specific base version."""
    task_id: str = Field(..., min_length=1, max_length=64)
    type: str = Field(..., pattern=r"^(outline|content)$")
    base_version: int = Field(..., ge=1)
    feedback: str = Field(default="", max_length=5000)


class VersionCleanRequest(BaseModel):
    """Clean old versions, keeping the latest N."""
    task_id: str = Field(..., min_length=1, max_length=64)
    keep_last: int = Field(default=20, ge=3, le=500)


# ── KB ────────────────────────────────────────────────────────────────────

class KBQueryRequest(BaseModel):
    """Query the knowledge base."""
    kb: str = Field(default="default", max_length=128)
    question: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(default=5, ge=1, le=50)
    filters: dict[str, Any] = Field(default_factory=dict)


class KBDeleteRequest(BaseModel):
    """Delete a KB or KB document."""
    kb: str = Field(..., max_length=128)
    doc_id: str | None = Field(default=None, max_length=256)


class TemplateSaveRequest(BaseModel):
    """Save a custom template."""
    name: str = Field(..., min_length=1, max_length=256, pattern=r"^[a-zA-Z0-9_\-一-鿿]+$")
    content: str = Field(..., max_length=50000)


# ── Section Editing ──────────────────────────────────────────────────────

class SectionEditRequest(BaseModel):
    """User edits a section of the generated content."""
    task_id: str = Field(..., min_length=1, max_length=64)
    section_name: str = Field(..., min_length=1, max_length=500)
    edited_content: str = Field(..., max_length=50000)
    mode: str = Field(default="edit", pattern=r"^(edit|regen)$")
