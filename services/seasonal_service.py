from __future__ import annotations

import os
import time
import json
import datetime as dt
from typing import Dict, Any, List, Optional

import requests
from geopy.geocoders import Nominatim
from services.ml_utils import log_training_data  # ✅ ML logging hook

# Optional: load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -------------------------------------------------
# API keys / config
# -------------------------------------------------
OPENTRIPMAP_API_KEY  = os.getenv("OPENTRIPMAP_KEY", "")
CALENDARIFIC_API_KEY = os.getenv("CALENDARIFIC_KEY", "")
PREDICTHQ_API_KEY    = os.getenv("PREDICTHQ_KEY", "")
UNSPLASH_ACCESS_KEY  = os.getenv("UNSPLASH_KEY", "")

# LLMs (optional)
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

USE_LLM = os.getenv("USE_LLM", "true").lower() in ("1", "true", "yes")

# Lazy provider detection
_HAVE_GEMINI = False
_HAVE_OPENAI = False
try:
    import google.generativeai as genai
    _HAVE_GEMINI = True
except Exception:
    pass

try:
    from openai import OpenAI
    _HAVE_OPENAI = True
except Exception:
    pass


# -----------------------------
# Helpers
# -----------------------------
def _season_from_date_str(date_str: Optional[str]) -> str:
    """Return 'winter'|'spring'|'summer'|'autumn'."""
    if not date_str:
        month = dt.date.today().month
    else:
        try:
            month = dt.date.fromisoformat(date_str).month
        except Exception:
            month = dt.date.today().month

    if month in (12, 1, 2): return "winter"
    if month in (3, 4, 5): return "spring"
    if month in (6, 7, 8): return "summer"
    return "autumn"


def geocode_place(place: str) -> Dict[str, Any]:
    """Geocode with Nominatim."""
    geocoder = Nominatim(user_agent="seasonal_suggestions_app")
    loc = geocoder.geocode(place)
    if not loc:
        raise ValueError(f"Could not geocode destination: {place}")

    country_code = ""
    try:
        loc2 = geocoder.geocode(place, addressdetails=True)
        if loc2 and hasattr(loc2, "raw"):
            country_code = (loc2.raw.get("address", {}).get("country_code", "") or "").upper()
    except Exception:
        pass

    return {
        "lat": loc.latitude,
        "lon": loc.longitude,
        "address": loc.address,
        "country_code": country_code,
    }


def _unsplash_image(query: str) -> Optional[str]:
    """Return Unsplash image URL."""
    if not UNSPLASH_ACCESS_KEY:
        return None
    try:
        r = requests.get(
            "https://api.unsplash.com/search/photos",
            params={"query": query, "per_page": 1, "orientation": "landscape"},
            headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"},
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            if results:
                return results[0]["urls"]["regular"]
    except Exception:
        pass
    return None


# -----------------------------
# Third-party APIs
# -----------------------------
def fetch_attractions_opentripmap(lat: float, lon: float, radius: int = 8000, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch attractions near lat/lon."""
    if not OPENTRIPMAP_API_KEY:
        return []

    base = "https://api.opentripmap.com/0.1/en/places"
    try:
        r = requests.get(
            f"{base}/radius",
            params={"radius": radius, "lon": lon, "lat": lat, "format": "json", "limit": limit, "apikey": OPENTRIPMAP_API_KEY},
            timeout=15,
        )
        r.raise_for_status()
        items = r.json()
        features: List[Dict[str, Any]] = []

        for it in items:
            xid = it.get("xid")
            name = it.get("name") or ""
            if not xid or not name:
                continue

            time.sleep(0.12)
            d = requests.get(f"{base}/xid/{xid}", params={"apikey": OPENTRIPMAP_API_KEY}, timeout=12)
            if d.status_code != 200:
                continue
            detail = d.json()

            kinds = (detail.get("kinds") or "").split(",")
            desc = detail.get("wikipedia_extracts", {}).get("text") or detail.get("info", {}).get("descr") or ""
            img = detail.get("preview", {}).get("source") or _unsplash_image(name)

            features.append({
                "name": name,
                "kinds": kinds,
                "desc": desc,
                "lat": detail.get("point", {}).get("lat"),
                "lon": detail.get("point", {}).get("lon"),
                "image": img,
                "source": "opentripmap",
                "url": detail.get("otm"),
            })
        return features
    except Exception:
        return []


def fetch_holidays_calendarific(country_code: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """Fetch holidays between start/end."""
    if not CALENDARIFIC_API_KEY or not country_code:
        return []

    try:
        start = dt.date.fromisoformat(start_date) if start_date else dt.date.today().replace(month=1, day=1)
        end   = dt.date.fromisoformat(end_date)   if end_date   else dt.date.today().replace(month=12, day=31)
    except Exception:
        start = dt.date.today().replace(month=1, day=1)
        end   = dt.date.today().replace(month=12, day=31)

    years = sorted({start.year, end.year})
    results: List[Dict[str, Any]] = []

    try:
        for y in years:
            r = requests.get(
                "https://calendarific.com/api/v2/holidays",
                params={"api_key": CALENDARIFIC_API_KEY, "country": country_code, "year": y},
                timeout=12,
            )
            if r.status_code != 200:
                continue
            data = r.json().get("response", {}).get("holidays", [])
            for h in data:
                try:
                    d = dt.date.fromisoformat(h.get("date", {}).get("iso", "")[:10])
                except Exception:
                    continue
                if not (start <= d <= end):
                    continue
                results.append({
                    "name": h.get("name"),
                    "date": h.get("date", {}).get("iso"),
                    "description": h.get("description") or "",
                    "type": ", ".join(h.get("type") or []),
                    "source": "calendarific",
                    "image": _unsplash_image(h.get("name", "")),
                    "url": h.get("url") or "https://calendarific.com/",
                })
    except Exception:
        pass
    return results


def fetch_events_predicthq(lat: float, lon: float, start_date: str, end_date: str, limit: int = 12) -> List[Dict[str, Any]]:
    """Fetch events near lat/lon."""
    if not PREDICTHQ_API_KEY:
        return []
    try:
        r = requests.get(
            "https://api.predicthq.com/v1/events/",
            headers={"Authorization": f"Bearer {PREDICTHQ_API_KEY}"},
            params={
                "within": f"30km@{lat},{lon}",
                "start.gte": start_date or dt.date.today().isoformat(),
                "start.lte": end_date or (dt.date.today() + dt.timedelta(days=30)).isoformat(),
                "limit": limit,
                "sort": "start",
            },
            timeout=15,
        )
        if r.status_code != 200:
            return []
        data = r.json().get("results", [])
        results: List[Dict[str, Any]] = []
        for e in data:
            title = e.get("title") or "Event"
            results.append({
                "name": title,
                "date": e.get("start"),
                "description": e.get("description") or "",
                "type": ", ".join(e.get("labels") or []),
                "source": "predicthq",
                "image": _unsplash_image(title),
                "url": e.get("url") or "https://predicthq.com/",
            })
        return results
    except Exception:
        return []


# -----------------------------
# LLM enrichment
# -----------------------------
SYSTEM_PROMPT = """You are a travel-planning assistant.
Given JSON 'meta' and 'results' (attractions, events), you will:
- produce a short destination overview (2-4 sentences),
- extract 5 highlights (bulleted, human-friendly),
- propose a 3-day itinerary,
- suggest image keywords if missing.
Return valid JSON."""

def _enrich_with_llm(meta: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
    if not USE_LLM:
        return {}

    provider = None
    if GEMINI_API_KEY and _HAVE_GEMINI:
        provider = "gemini"
        genai.configure(api_key=GEMINI_API_KEY)
    elif OPENAI_API_KEY and _HAVE_OPENAI:
        provider = "openai"
    else:
        return {}

    user_payload = json.dumps(
        {"meta": meta, "results": {"attractions": results.get("attractions", []), "events": results.get("events", [])}},
        ensure_ascii=False,
    )

    try:
        if provider == "gemini":
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(
                [{"role": "user", "parts": [{"text": SYSTEM_PROMPT}, {"text": user_payload}]}]
            )
            text = resp.text or "{}"
        else:
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_payload},
                ],
            )
            text = resp.choices[0].message.content or "{}"

        data = json.loads(text)
        return {
            "overview": data.get("overview", ""),
            "highlights": data.get("highlights", []),
            "itinerary": data.get("itinerary", []),
            "image_hints": data.get("image_hints", []),
        }
    except Exception:
        return {}


# -----------------------------
# Public service function
# -----------------------------
def get_seasonal_suggestions(
    frm: Optional[str],
    to: str,
    start: Optional[str],
    end: Optional[str],
    use_llm: Optional[bool] = None,
) -> Dict[str, Any]:
    if not to:
        return {
            "meta": {},
            "results": {"attractions": [], "events": []},
            "enrichment": {},
            "status": "error",
            "message": "Destination (to) is required.",
        }

    try:
        season = _season_from_date_str(start)
        geo = geocode_place(to)
        lat, lon = geo["lat"], geo["lon"]
        country = geo.get("country_code", "")

        attractions = fetch_attractions_opentripmap(lat, lon)
        holidays = fetch_holidays_calendarific(country, start or "", end or "")
        events = fetch_events_predicthq(lat, lon, start or "", end or "")
        combined_events = holidays + events

        meta = {
            "destination": to,
            "lat": lat,
            "lon": lon,
            "country": country,
            "season": season,
            "from": frm or "",
            "start": start or "",
            "end": end or "",
        }

        results = {"attractions": attractions, "events": combined_events}
        status = "ok" if (attractions or combined_events) else "empty"
        message = "" if (attractions or combined_events) else f"No data found for {to}."

        # ✅ ML logging
        try:
            log_training_data(to, results, season, country)
        except Exception as e:
            print(f"❌ [LOGGING ERROR] {e}")

        # ✅ LLM enrichment
        allow_llm = USE_LLM if use_llm is None else bool(use_llm)
        enrichment = _enrich_with_llm(meta, results) if allow_llm else {}

        return {"meta": meta, "results": results, "enrichment": enrichment, "status": status, "message": message}

    except Exception as e:
        return {
            "meta": {"destination": to, "season": _season_from_date_str(start)},
            "results": {"attractions": [], "events": []},
            "enrichment": {},
            "status": "error",
            "message": str(e),
        }
