"""Utility for extracting KPIs with the OpenRouter API."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import re
import time
import random
from typing import Optional, List, Iterable

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    OpenAI,
)

from src import prompts

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

_ENV_LOADED = False


def _ensure_env_loaded() -> None:
    """Load environment variables from the project .env file once."""

    global _ENV_LOADED
    if _ENV_LOADED:
        return
    if load_dotenv is None:
        LOGGER.debug("python-dotenv nicht installiert – überspringe automatisches Laden der .env")
        _ENV_LOADED = True
        return
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        LOGGER.debug(".env-Datei aus %s geladen", env_path)
    else:
        LOGGER.debug("Keine .env-Datei unter %s gefunden", env_path)
    _ENV_LOADED = True


@dataclass
class FinancialKPI:
    """Structured container for extracted financial metrics."""

    kpi: str
    value: str
    unit: str
    context: str
    kategorie: str | None = None       # NEU: KPI-Kategorie aus dem LLM
    reportingyear: str | None = None   # Jahr des Ursprungsberichts (wird außerhalb gesetzt)


class OpenRouterClient:
    """Minimal client for calling the OpenRouter chat completions API."""

    def __init__(self) -> None:
        _ensure_env_loaded()
        try:
            self.api_key = os.environ["OPENROUTER_API_KEY"]
            self.base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            self.model = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-70b-instruct")
            self.site_url = os.environ.get("OPENROUTER_SITE_URL")
            self.site_name = os.environ.get("OPENROUTER_SITE_NAME")
        except KeyError as exc:  # pragma: no cover - configuration guard
            raise RuntimeError("OpenRouter credentials missing") from exc

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _build_headers(self) -> dict:
        headers: dict = {}
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name
        return headers

    def complete(self, messages: List[dict]) -> str:
        """Send a chat completion request and return the text response."""

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_headers=self._build_headers(),
                extra_body={},
            )
        except APIStatusError as exc:
            if exc.status_code == 401:
                LOGGER.error(
                    "OpenRouter-Antwort 401 Unauthorized – bitte API-Schlüssel und Tarif prüfen"
                )
                raise RuntimeError("OpenRouter-Zugriff verweigert (401 Unauthorized)") from exc
            raise RuntimeError(f"OpenRouter-Anfrage fehlgeschlagen: {exc}") from exc
        except (APIConnectionError, APIError) as exc:  # pragma: no cover - network guard
            raise RuntimeError("Verbindung zu OpenRouter fehlgeschlagen") from exc

        try:
            return completion.choices[0].message.content
        except (AttributeError, IndexError, KeyError) as exc:  # pragma: no cover - safety
            raise RuntimeError("Antwort von OpenRouter unvollständig") from exc


# Sinnvolle Defaults für OpenRouter Free-Tier (aus Logs ersichtlich)
REQS_PER_MIN_DEFAULT = 16      # X-RateLimit-Limit: 16/min
SAFETY_MARGIN = 0.90           # etwas unter Limit bleiben
MIN_SLEEP_BETWEEN_REQ = 4.2    # ~ 60/14 ≈ 4.2s (unter 16/min bleiben)

# Anzahl Chunks pro Batch: konservativ halten (Free-Modelle)
BATCH_SIZE = 8


def _strip_code_fences(s: str) -> str:
    """Entfernt Markdown-Code-Fences und leading labels (```json ... ```)."""
    s = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s.strip())
    return s.strip()


def _extract_json_fragment(s: str) -> Optional[str]:
    """
    Versucht, aus einer Antwort den JSON-Array-Teil robust zu extrahieren.
    - Schneidet auf Bereich zwischen erstem '[' und letztem ']' zu.
    - Entfernt offenkundige Trunkierungen am Ende.
    """
    s = _strip_code_fences(s)
    start = s.find("[")
    end = s.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    fragment = s[start : end + 1]
    # Versuch, triviale JSON-Fehler zu entschärfen: trailing-Kommas entfernen
    fragment = re.sub(r",\s*([}\]])", r"\1", fragment)
    return fragment


def _parse_json_lenient(raw: str) -> Optional[list]:
    """
    Tolerantes JSON-Parsing:
    1) Direkter json.loads-Versuch.
    2) Falls Fehler: Array-Fragment extrahieren und erneut laden.
    3) Falls weiterhin Fehler: triviale Klammer-Reparaturen.
    """
    try:
        return json.loads(_strip_code_fences(raw))
    except json.JSONDecodeError:
        pass

    frag = _extract_json_fragment(raw)
    if frag:
        try:
            return json.loads(frag)
        except json.JSONDecodeError:
            repaired = frag
            # naive Reparatur: fehlende geschweifte/eckige Klammern ergänzen
            if repaired.count("{") > repaired.count("}"):
                repaired += "}" * (repaired.count("{") - repaired.count("}"))
            if repaired.count("[") > repaired.count("]"):
                repaired += "]" * (repaired.count("[") - repaired.count("]"))
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                return None
    return None


def _contains_numbers(text: str) -> bool:
    """
    Prüft heuristisch, ob ein Textabschnitt numerische Zeichen enthält.
    Wird als Vorfilter genutzt, um nur potenziell KPI-relevante Chunks
    an die API zu senden.
    """
    return bool(re.search(r"\d", text))


def _respect_minute_budget(last_call_ts: list, per_min_limit: int = REQS_PER_MIN_DEFAULT) -> None:
    """
    Erzwingt Pacing zwischen Requests, sodass wir unter dem pro-Minuten-Limit bleiben.
    last_call_ts ist eine 1-Element-Liste (mutable Timestamp), um Aufrufzeitpunkte zu teilen.
    """
    now = time.time()
    if last_call_ts[0] is not None:
        elapsed = now - last_call_ts[0]
        min_interval = max(MIN_SLEEP_BETWEEN_REQ, 60.0 / (per_min_limit * SAFETY_MARGIN))
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
    last_call_ts[0] = time.time()


def _complete_with_retries(client: OpenRouterClient, messages: List[dict],
                           max_retries: int = 5) -> str:
    """
    Ruft das Modell mit exponentiellem Backoff und Jitter auf.
    Handhabt 429/5xx-Fehler robust; gibt bei persistenten Fehlern auf.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return client.complete(messages)
        except RuntimeError as exc:
            msg = str(exc)
            # 429 → Backoff
            if "429" in msg or "rate limit" in msg.lower():
                wait = min(30.0, (2 ** attempt) + random.uniform(0, 1.0))
                LOGGER.warning("Rate-Limit (%s). Warte %.1fs (Versuch %d/%d).",
                               msg, wait, attempt, max_retries)
                time.sleep(wait)
                continue
            # vorübergehende Serverfehler
            if "502" in msg or "503" in msg or "504" in msg:
                wait = min(20.0, (2 ** attempt) + random.uniform(0, 1.0))
                LOGGER.warning("Upstream-Fehler (%s). Warte %.1fs (Versuch %d/%d).",
                               msg, wait, attempt, max_retries)
                time.sleep(wait)
                continue
            # andere Fehler → sofort weiterreichen
            raise
    raise RuntimeError("Maximale Retry-Versuche erreicht")


def extract_financial_kpis(chunks: Iterable[str]) -> List[FinancialKPI]:
    """
    Extrahiert finanzielle Kennzahlen abschnittsweise:
    - Numerischer Vorfilter: nur Chunks mit mindestens einer Ziffer werden betrachtet
    - Batching von Chunks (BATCH_SIZE)
    - Pacing unterhalb der Free-Tier-Limits (REQS_PER_MIN)
    - Exponentielles Backoff bei 429/5xx
    - Tolerantes JSON-Parsing
    - Inkrementelle Aggregation
    """
    client = OpenRouterClient()
    all_kpis: List[FinancialKPI] = []

    # Ursprüngliche Chunk-Liste materialisieren
    original_chunks = list(chunks)

    # Numerischer Vorfilter: Chunks ohne Ziffern werden nicht an die API gesendet
    chunk_list = [c for c in original_chunks if _contains_numbers(c)]
    filtered_out = len(original_chunks) - len(chunk_list)

    LOGGER.info(
        "Numerischer Vorfilter aktiv: %d von %d Chunks enthalten mindestens eine Ziffer; "
        "%d Chunks werden übersprungen.",
        len(chunk_list),
        len(original_chunks),
        filtered_out,
    )

    n = len(chunk_list)
    LOGGER.info("Starte KPI-Extraktion für %d vorgefilterte Textabschnitte", n)

    # Falls nach Filterung keine Chunks übrig bleiben, können wir direkt abbrechen
    if n == 0:
        LOGGER.info("Keine Chunks mit numerischen Inhalten gefunden – keine KPI-Extraktion durchgeführt.")
        return all_kpis

    last_call_ts = [None]  # mutable Timestamp für Pacing

    for i in range(0, n, BATCH_SIZE):
        batch = chunk_list[i : i + BATCH_SIZE]
        batch_text = "\n\n".join(batch)

        LOGGER.info("Sende Batch %d–%d an OpenRouter", i + 1, min(i + BATCH_SIZE, n))

        messages = [
            {
                "role": "system",
                "content": prompts.KPI_EXTRACTION_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    "Analysiere den folgenden Abschnitt eines Geschäftsberichts strikt gemäß den "
                    "obigen Anweisungen. Extrahiere alle relevanten Kennzahlen und gib sie als "
                    "JSON-Liste mit Objekten der Form "
                    '{"kpi": "...", "wert": ..., "einheit": "...", "kontext": "...", "kategorie": "..."} '
                    "zurück. Wenn keine relevanten Kennzahlen vorkommen, gib [] zurück.\n\n"
                    f"{batch_text}"
                ),
            },
        ]

        # pro-Minuten-Limit respektieren (vor Anfrage)
        _respect_minute_budget(last_call_ts, per_min_limit=REQS_PER_MIN_DEFAULT)

        try:
            raw_response = _complete_with_retries(client, messages, max_retries=5)
        except RuntimeError as exc:
            LOGGER.error("KPI-Extraktion für Batch %d abgebrochen: %s", i, exc)
            continue

        parsed = _parse_json_lenient(raw_response)
        if parsed is None:
            LOGGER.warning("LLM-Antwort war kein brauchbares JSON (Batch %d): %s",
                           i, raw_response[:600])
            continue

        # Normalisierung & Sammeln
        # Normalisierung & Sammeln
        for entry in parsed:
            k = (entry.get("kpi", "") or "").strip()
            v = (str(entry.get("wert", "")) or "").strip()
            u = (entry.get("einheit", "") or "").strip()
            c = (entry.get("kontext", "") or "").strip()
            cat = (entry.get("kategorie", "") or "").strip()

            if not k:
                continue

            all_kpis.append(
                FinancialKPI(
                    kpi=k,
                    value=v,
                    unit=u,
                    context=c,
                    kategorie=cat or None,
                )
            )

        LOGGER.info("Batch %d verarbeitet – kumuliert %d KPIs",
                    (i // BATCH_SIZE) + 1, len(all_kpis))

    LOGGER.info("Gesamte KPI-Extraktion abgeschlossen (%d KPIs gefunden)", len(all_kpis))
    return all_kpis
