import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field


APP_DIR = Path(__file__).resolve().parent
KNOWLEDGE_PATH = APP_DIR / "knowledge_pack.md"

app = FastAPI(title="LEGACY Engine", version="0.1.0")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


class SparkRequest(BaseModel):
    spark: str = Field(min_length=10, description="Luke's dictated Daily Spark.")
    source_title: str | None = None


class ContentPackage(BaseModel):
    title: str
    central_idea: str
    recommended_primary_platform: str
    linkedin: str
    instagram: str
    facebook: str
    x: str
    image_idea: str
    hashtags: list[str]
    risk_warning: str | None = None


def load_knowledge_pack() -> str:
    if not KNOWLEDGE_PATH.exists():
        raise RuntimeError("knowledge_pack.md is missing.")
    return KNOWLEDGE_PATH.read_text(encoding="utf-8")


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise ValueError("Model did not return valid JSON.") from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "legacy-engine"}


@app.post("/generate", response_model=ContentPackage)
def generate_content(
    request: SparkRequest,
    x_legacy_secret: str | None = Header(default=None),
) -> ContentPackage:
    expected_secret = os.environ.get("LEGACY_WEBHOOK_SECRET")
    if expected_secret and x_legacy_secret != expected_secret:
        raise HTTPException(status_code=401, detail="Invalid webhook secret.")

    knowledge = load_knowledge_pack()
    model = os.environ.get("OPENAI_MODEL", "gpt-5.5")

    instructions = f"""
You are the LEGACY Content Engine for Luke Fedlam.

Your only job is to transform a Daily Spark into publication-ready social media drafts.
Do not brainstorm beyond the request. Do not recommend a keynote, workshop, book,
framework, or product unless specifically asked. Default to execution.

Use the following shared knowledge and voice rules:

--- BEGIN LEGACY KNOWLEDGE PACK ---
{knowledge}
--- END LEGACY KNOWLEDGE PACK ---

Return ONLY valid JSON matching this exact structure:
{{
  "title": "short descriptive title",
  "central_idea": "one sentence",
  "recommended_primary_platform": "LinkedIn, Instagram, Facebook, or X",
  "linkedin": "publication-ready LinkedIn post",
  "instagram": "publication-ready Instagram caption",
  "facebook": "publication-ready Facebook post",
  "x": "one publication-ready X post, not a thread",
  "image_idea": "one concise visual recommendation",
  "hashtags": ["maximum", "five", "hashtags"],
  "risk_warning": null
}}

Rules:
- Identify ONE strongest idea.
- Write like Luke speaks: positive, energetic, passionate, caring, practical, and impactful.
- Start with the strongest human hook, not background information.
- Preserve Luke's substance and conviction.
- Adapt for each platform; do not copy the same text four times.
- Never sound salesy, preachy, overly lawyerly, generic, or AI-generated.
- Never invent facts, quotations, audience reactions, or client details.
- Flag confidentiality, legal-risk, political, or factual-verification concerns in risk_warning.
- Keep hashtags to five or fewer.
"""

    user_input = request.spark
    if request.source_title:
        user_input = f"Source title: {request.source_title}\n\nDaily Spark:\n{request.spark}"

    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=user_input,
    )

    try:
        payload = extract_json(response.output_text)
        return ContentPackage.model_validate(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not validate model output: {exc}",
        ) from exc
