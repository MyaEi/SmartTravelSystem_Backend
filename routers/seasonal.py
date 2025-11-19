from fastapi import APIRouter, Query
from services.seasonal_service import get_seasonal_suggestions

router = APIRouter(prefix="", tags=["seasonal"])

@router.get("/suggestions")
def suggestions(
    frm: str | None = Query(None),
    to: str = Query(..., description="Destination city or place"),
    start: str | None = Query(None),
    end: str | None = Query(None),
    use_llm: bool = Query(True, description="Enable LLM enrichment")
):
    data = get_seasonal_suggestions(frm, to, start, end, use_llm)

    meta = data.get("meta", {})
    results = data.get("results", {})
    enrichment = data.get("enrichment", {})
    status = data.get("status", "ok")
    message = data.get("message", "")

    return {
        "meta": meta,
        "results": results,
        "enrichment": enrichment,
        "status": status,
        "message": message,
    }
