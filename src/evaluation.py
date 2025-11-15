"""Greenwashing evaluation using the OpenRouter LLM."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Dict, Iterable

from src import prompts
from src.kpi_extraction import FinancialKPI, OpenRouterClient

LOGGER = logging.getLogger(__name__)


def analyse_greenwashing(claims: Iterable[object], kpis: Iterable[FinancialKPI]) -> Dict[str, object]:
    """
    Bewertet Greenwashing-Risiko durch Abgleich von Nachhaltigkeitsclaims und Finanz-KPIs.
    Übergibt Claims inklusive p_yes (falls vorhanden) sowie die KPIs als JSON an das LLM.
    """

    client = OpenRouterClient()

    # Claims in strukturiertes JSON überführen: text + p_yes (falls vorhanden)
    claims_list = []
    for c in claims:
        # Robust auf Attribute/Keys prüfen (unterstützt Dataclass, NamedTuple, dict)
        text = getattr(c, "text", None)
        if text is None and isinstance(c, dict):
            text = c.get("text") or c.get("sentence")

        # p_yes kann als prob_yes im Claim liegen (gemappt) – ansonsten None
        p_yes = None
        if hasattr(c, "prob_yes"):
            p_yes = getattr(c, "prob_yes")
        elif isinstance(c, dict):
            p_yes = c.get("prob_yes") or c.get("p_yes")

        entry = {"text": text}
        if p_yes is not None:
            entry["p_yes"] = p_yes

        # Nur valide Claims übernehmen
        if entry.get("text"):
            claims_list.append(entry)

    # KPIs als strukturierte JSON-Liste
    kpis_list = [asdict(k) for k in kpis]

    # Kompakter, deterministischer Prompt: rein strukturierte Nutzlast
    payload = {
        "claims": claims_list,     # [{ "text": "...", "p_yes": 0.98 }, ...]
        "kpis": kpis_list,         # [{ "kpi": "...", "value": "...", "unit": "...", "context": "..." }, ...]
    }

    user_prompt = json.dumps(payload, ensure_ascii=False)

    messages = [
        {"role": "system", "content": prompts.GREENWASHING_ANALYSIS_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw_response = client.complete(messages)
    except RuntimeError as exc:
        LOGGER.error("Greenwashing-Bewertung fehlgeschlagen: %s", exc)
        return {
            "verdaechtiger_score": None,
            "einschaetzung": "Bewertung fehlgeschlagen",
            "rationales": [str(exc)],
        }

    # Erwartet wird JSON vom LLM
    try:
        result = json.loads(raw_response)
    except json.JSONDecodeError:
        LOGGER.warning("LLM returned invalid JSON: %s", raw_response)
        return {
            "verdaechtiger_score": None,
            "einschaetzung": "Bewertung fehlgeschlagen",
            "rationales": [raw_response],
        }

    return result