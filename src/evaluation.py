"""Bewertet Claim-KPI-Zusammenhänge mithilfe des OpenRouter-LLM."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Any, Optional

from src import prompts
from src.kpi_extraction import FinancialKPI, OpenRouterClient
import re

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hilfsfunktionen für robustes JSON-Parsing (unverändert)
# ---------------------------------------------------------------------------

def _strip_code_fences_obj(s: str) -> str:
    """Entfernt Markdown-Code-Fences und führende ```json-Labels."""
    s = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s.strip())
    return s.strip()


def _extract_json_object(s: str) -> Optional[str]:
    """
    Versucht, aus einer Antwort den JSON-Objekt-Teil robust zu extrahieren.
    - Schneidet auf Bereich zwischen erstem '{' und letztem '}' zu.
    Hinweis: Funktioniert primär für Objekte; bei Listen sollte idealerweise
    der erste json.loads()-Versuch schon greifen.
    """
    s = _strip_code_fences_obj(s)
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    fragment = s[start: end + 1]
    # einfache Reparatur: trailing-Kommas vor '}' oder ']' entfernen
    fragment = re.sub(r",\s*([}\]])", r"\1", fragment)
    return fragment


def _parse_json_lenient_object(raw: str) -> Optional[Any]:
    """
    Tolerantes JSON-Parsing.
    Versucht zunächst, die LLM-Antwort direkt als JSON (Objekt ODER Liste)
    zu interpretieren. Falls das fehlschlägt, wird ein Objektfragment
    extrahiert und erneut versucht.
    """
    try:
        return json.loads(_strip_code_fences_obj(raw))
    except json.JSONDecodeError:
        pass

    frag = _extract_json_object(raw)
    if frag:
        try:
            return json.loads(frag)
        except json.JSONDecodeError:
            return None

    return None


# ---------------------------------------------------------------------------
# Alte Evaluation (Claims + KPIs direkt)
# ---------------------------------------------------------------------------

def analyse_greenwashing(claims: Iterable[object], kpis: Iterable[FinancialKPI]) -> Dict[str, object]:
    """
    Ursprüngliche Greenwashing-Bewertung auf Basis von Claims + KPIs.
    Kann weiterhin für Experimente genutzt werden, ist aber NICHT
    batch-basiert und nutzt NICHT die Matching-Dateien.
    """

    client = OpenRouterClient()

    # Claims in strukturiertes JSON überführen: text + p_yes
    claims_list = []
    for c in claims:
        text = getattr(c, "text", None)
        if text is None and isinstance(c, dict):
            text = c.get("text") or c.get("sentence")

        p_yes = None
        if hasattr(c, "prob_yes"):
            p_yes = getattr(c, "prob_yes")
        elif isinstance(c, dict):
            p_yes = c.get("prob_yes") or c.get("p_yes")

        entry = {"text": text}
        if p_yes is not None:
            entry["p_yes"] = p_yes

        if entry.get("text"):
            claims_list.append(entry)

    kpis_list = [asdict(k) for k in kpis]

    payload = {
        "claims": claims_list,
        "kpis": kpis_list,
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

    result = _parse_json_lenient_object(raw_response)
    if result is None:
        LOGGER.warning("LLM returned invalid or non-JSON response: %s", raw_response[:800])
        return {
            "verdaechtiger_score": 0.0,
            "einschaetzung": "Technischer Fehler: LLM-Antwort war kein gültiges JSON.",
            "rationales": [
                "Claim: technische Verarbeitung | KPI: LLM-Output | Schlussfolgerung: Antwort war kein gültiges JSON, Fallback verwendet."
            ],
            "evidence": {
                "claim_to_kpi_links": []
            },
        }
    return result


# ---------------------------------------------------------------------------
# Batch-basierte Evaluation auf Basis der Matching-Datei
# ---------------------------------------------------------------------------

def _prepare_batch_payload_from_matching_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrahiert aus einem Matching-Batch die für das LLM relevante Struktur:
    - 'claims': Liste von { 'text': ..., 'p_yes': ... (optional) }
    - 'kpis':   Liste von KPI-Objekten (wird direkt durchgereicht)

    Die Matching-Batch-Struktur stammt aus den Dateien:
    {
      "batch_id": 0,
      "topics": [...],
      "claims": [
        { "claim_id": "...", "text": "...", "p_yes": ..., "label": "YES" },
        ...
      ],
      "kpis": [
        { "kpi": "...", "value": "...", "unit": "...", "context": "...", ... },
        ...
      ],
      "meta": {...}
    }
    """
    claims_raw = batch.get("claims", [])
    kpis_raw = batch.get("kpis", [])

    claims_list: List[Dict[str, Any]] = []
    for c in claims_raw:
        text = c.get("text")
        if not text:
            continue
        entry: Dict[str, Any] = {"text": text}
        if "p_yes" in c and c["p_yes"] is not None:
            entry["p_yes"] = c["p_yes"]
        claims_list.append(entry)

    # KPIs können direkt übernommen werden; das Prompt erwartet mindestens
    # 'kpi', 'value', 'unit', 'context', zusätzliche Felder sind unkritisch.
    kpis_list: List[Dict[str, Any]] = list(kpis_raw)

    return {
        "claims": claims_list,
        "kpis": kpis_list,
    }


def analyse_greenwashing_batch_from_matching_batch(
    batch: Dict[str, Any],
    client: Optional[OpenRouterClient] = None,
) -> List[Dict[str, Any]]:
    """
    Führt die Greenwashing-Bewertung für GENAU EINEN Batch aus einer
    Matching-Datei durch.

    Rückgabe: Liste von Objekten der Form
        {
          "claim": "<Claim-Text>",
          "kpis": ["<KPI-String 1>", "<KPI-String 2>", ...],
          "relation": "stutzt|widerspricht|kein_beleg",
          "rationale": "<Begründung>"
        }
    entsprechend GREENWASHING_ANALYSIS_PROMPT.
    """
    if client is None:
        client = OpenRouterClient()

    payload = _prepare_batch_payload_from_matching_batch(batch)

    # Falls Batch keine Claims enthält, leere Liste zurückgeben
    if not payload["claims"]:
        return []

    user_prompt = json.dumps(payload, ensure_ascii=False)
    messages = [
        {"role": "system", "content": prompts.GREENWASHING_ANALYSIS_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw_response = client.complete(messages)
    except RuntimeError as exc:
        LOGGER.error(
            "Greenwashing-Bewertung für Batch %s fehlgeschlagen: %s",
            batch.get("batch_id"),
            exc,
        )
        # Fallback: für jeden Claim ein technischer Kein-Beleg-Eintrag
        results: List[Dict[str, Any]] = []
        for c in payload["claims"]:
            results.append({
                "claim": c["text"],
                "kpis": [],
                "relation": "kein_beleg",
                "rationale": "Technischer Fehler bei der LLM-Auswertung; keine Bewertung möglich.",
            })
        return results

    parsed = _parse_json_lenient_object(raw_response)
    if parsed is None:
        LOGGER.warning(
            "LLM returned invalid or non-JSON response for batch %s: %s",
            batch.get("batch_id"),
            raw_response[:800],
        )
        results: List[Dict[str, Any]] = []
        for c in payload["claims"]:
            results.append({
                "claim": c["text"],
                "kpis": [],
                "relation": "kein_beleg",
                "rationale": "Die LLM-Antwort war kein gültiges JSON; Claim konnte nicht valide bewertet werden.",
            })
        return results

    if not isinstance(parsed, list):
        LOGGER.warning(
            "LLM-Antwort für Batch %s war kein JSON-Array, sondern Typ %s – wird verworfen.",
            batch.get("batch_id"),
            type(parsed),
        )
        results: List[Dict[str, Any]] = []
        for c in payload["claims"]:
            results.append({
                "claim": c["text"],
                "kpis": [],
                "relation": "kein_beleg",
                "rationale": "Die LLM-Antwort entsprach nicht dem erwarteten Listenformat.",
            })
        return results

    # Defensive Normalisierung der Einträge
    normalized: List[Dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue

        claim_text = item.get("claim")
        relation = item.get("relation")
        rationale = item.get("rationale")

        # kpis kann Liste sein, None oder ein einzelner String
        raw_kpis = item.get("kpis", [])
        if raw_kpis is None:
            raw_kpis = []
        if not isinstance(raw_kpis, list):
            raw_kpis = [raw_kpis]

        # alles zu Strings normalisieren
        kpis_norm = [str(k) for k in raw_kpis if k is not None]

        if claim_text is None or relation not in ("stutzt", "widerspricht", "kein_beleg"):
            # überspringen, wenn die Kernstruktur fehlt
            continue

        normalized.append(
            {
                "claim": claim_text,
                "kpis": kpis_norm,
                "relation": relation,
                "rationale": rationale or "",
            }
        )

    return normalized


def analyse_matching_file(
    matching_path: Path,
    client: Optional[OpenRouterClient] = None,
) -> List[Dict[str, Any]]:
    """
    Liest eine Matching-Datei ein und führt für jeden Batch eine LLM-Auswertung durch.

    Erwartete Matching-Dateistruktur:
    {
      "company_key": "...",
      "claims_file": "...",
      "kpi_files": [...],
      "batches": [
        {
          "batch_id": 0,
          "topics": [...],
          "claims": [...],
          "kpis": [...],
          "meta": {...}
        },
        ...
      ]
    }

    Rückgabe:
        Eine EINZIGE Liste über ALLE Batches mit Einträgen der Form:
        {
          "claim": "...",
          "kpis": ["<KPI-String 1>", "..."],
          "relation": "stutzt|widerspricht|kein_beleg",
          "rationale": "..."
        }
    """
    if client is None:
        client = OpenRouterClient()

    raw = matching_path.read_text(encoding="utf-8")
    data = json.loads(raw)

    batches = data.get("batches", [])
    company_key = data.get("company_key")
    claims_file = data.get("claims_file")

    LOGGER.info(
        "Starte LLM-Evaluation für Matching-Datei %s (company_key=%s, claims_file=%s) mit %d Batches.",
        matching_path.name,
        company_key,
        claims_file,
        len(batches),
    )

    all_results: List[Dict[str, Any]] = []

    for batch in batches:
        batch_id = batch.get("batch_id")
        LOGGER.info("Verarbeite Batch %s ...", batch_id)
        batch_results = analyse_greenwashing_batch_from_matching_batch(
            batch=batch,
            client=client,
        )
        all_results.extend(batch_results)

    return all_results
