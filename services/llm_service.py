# services/llm_service.py
import os
import json
from typing import Dict, List, Optional

# Optional providers — install only what you use:
# pip install google-generativeai openai
_PROVIDER = None
_HAVE_GEMINI = False
_HAVE_OPENAI = False

try:
    import google.generativeai as genai  # type: ignore
    _HAVE_GEMINI = True
except Exception:
    pass

try:
    from openai import OpenAI  # type: ignore
    _HAVE_OPENAI = True
except Exception:
    pass


def _pick_provider():
    """Choose the first enabled provider by env key."""
    if os.getenv("GEMINI_API_KEY") and _HAVE_GEMINI:
        return "gemini"
    if os.getenv("OPENAI_API_KEY") and _HAVE_OPENAI:
        return "openai"
    return None


def _ensure_init():
    global _PROVIDER
    if _PROVIDER is not None:
        return _PROVIDER
    _PROVIDER = _pick_provider()
    if _PROVIDER == "gemini":
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    elif _PROVIDER == "openai":
        # nothing else needed — client created per call
        pass
    return _PROVIDER


SYSTEM_PROMPT = """You are a travel-planning assistant.
Given JSON 'meta' and 'results' (attractions, events), you will:
- produce a short destination overview (2-4 sentences),
- extract 5 highlights (bulleted, human-friendly),
- propose a 3-day itinerary with morning/afternoon/evening blocks,
- suggest image keywords for each item if image_url is missing (2-4 words),
- keep dates factual; do not hallucinate new events.

Return strictly valid JSON matching this schema:
{
  "overview": "string",
  "highlights": ["string", "..."],
  "itinerary": [
    {"day": 1, "morning": "string", "afternoon": "string", "evening": "string"},
    {"day": 2, "morning": "string", "afternoon": "string", "evening": "string"},
    {"day": 3, "morning": "string", "afternoon": "string", "evening": "string"}
  ],
  "image_hints": [
    {"name": "item name", "keywords": "foo bar"},
    ...
  ]
}
If information is missing, be conservative. Use British date format only if meta.country == 'GB', otherwise ISO yyyy-mm-dd in text."""

def build_user_payload(meta: Dict, results: Dict) -> str:
    payload = {
        "meta": meta,
        "results": {
            "attractions": results.get("attractions", []),
            "events": results.get("events", []),
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def enrich_with_llm(meta: Dict, results: Dict) -> Optional[Dict]:
    """
    Returns an enrichment dict or None if no provider key was configured.
    """
    provider = _ensure_init()
    if provider is None:
        return None

    user_json = build_user_payload(meta, results)

    if provider == "gemini":
        model = genai.GenerativeModel("gemini-2.0-flash")
        resp = model.generate_content(
            [{"role": "user", "parts": [{"text": SYSTEM_PROMPT}, {"text": user_json}]}]
        )
        text = resp.text or "{}"

    elif provider == "openai":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Use a small, inexpensive model name you have access to
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.3,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_json},
            ],
        )
        text = resp.choices[0].message.content or "{}"

    try:
        data = json.loads(text)
        # light validation/fallbacks
        data.setdefault("overview", "")
        data.setdefault("highlights", [])
        data.setdefault("itinerary", [])
        data.setdefault("image_hints", [])
        return data
    except Exception:
        # If the model returned non-JSON, just skip enrichment
        return None
