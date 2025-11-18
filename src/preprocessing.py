"""Utilities for loading, cleaning and chunking report texts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

import pdfplumber
import re
from typing import List
import nltk
from nltk.tokenize import sent_tokenize

LOGGER = logging.getLogger(__name__)


def extract_text_from_pdfs(pdf_paths: Iterable[Path]) -> str:
    """Extract text from a collection of PDF files.

    Parameters
    ----------
    pdf_paths:
        Collection of paths pointing to PDF documents. The function ignores
        missing files gracefully while logging a warning. Extracted text is
        concatenated in the order the paths are provided.
    """

    texts: List[str] = []
    for path in pdf_paths:
        try:
            with pdfplumber.open(path) as pdf:
                pages = [page.extract_text(x_tolerance=2) or "" for page in pdf.pages]
                texts.append("\n".join(pages))
        except FileNotFoundError:
            LOGGER.warning("PDF not found: %s", path)
        except Exception as exc:  # pragma: no cover - defensive branch
            LOGGER.error("Failed to parse %s: %s", path, exc)
    return "\n".join(texts)

def clean_text(text: str) -> str:
    """Perform enhanced cleaning to ease downstream processing.

    - Normalisiert Leerzeichen
    - Entfernt typische Bullet-/Icon-Symbole (z.B. , •)
    - Führt Zeilen zusammen, wenn sie wahrscheinlich zu einem Satz gehören
    """

    if not text:
        return ""

    # 1) Grundnormalisierung: NBSP -> Space
    cleaned = text.replace("\u00a0", " ")

    # 2) Typische Bullet-/Icon-Symbole entfernen
    #    (Liste bei Bedarf erweitern, je nach PDFs)
    cleaned = re.sub(r"[•●■▪◦▶►▸▹]", " ", cleaned)

    # 3) Zeilenbasis: Trim + leere Zeilen entfernen
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]

    merged_lines = []
    buffer = ""

    # Heuristik:
    # - Wenn eine Zeile nicht mit Satzzeichen (.?!:) endet,
    #   wird sie mit der nächsten Zeile verbunden (sofern diese nicht wie eine Überschrift in GROSSBUCHSTABEN aussieht).
    heading_pattern = re.compile(r"^[A-Z][A-Z0-9\s\-\&]{3,}$")

    for line in lines:
        if not buffer:
            buffer = line
            continue

        # Prüfe, ob buffer wahrscheinlich ein vollständiger Satz/Absatz ist
        ends_with_punct = bool(re.search(r"[\.!?;:]$", buffer))

        # Prüfe, ob aktuelle Zeile eher eine Überschrift in Capitals ist
        looks_like_heading = bool(heading_pattern.match(line))

        if not ends_with_punct and not looks_like_heading:
            # Wir hängen die aktuelle Zeile an den vorherigen Buffer an
            buffer = buffer + " " + line
        else:
            # buffer ist abgeschlossen, wir starten eine neue Einheit
            merged_lines.append(buffer)
            buffer = line

    if buffer:
        merged_lines.append(buffer)

    # 4) Finale Normalisierung der Whitespaces
    cleaned = "\n".join(merged_lines)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n") if line.strip())

    return cleaned


def chunk_text(text: str, *, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into slightly overlapping segments.

    Parameters
    ----------
    text:
        Cleaned input text.
    chunk_size:
        Maximum length of a chunk measured in tokens approximated by number of
        words. The heuristic keeps the implementation lightweight while still
        being practical for prototyping.
    overlap:
        Number of words shared between neighbouring chunks to maintain context.
    """

    if not text:
        return []

    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += max(chunk_size - overlap, 1)
    return chunks

def sentences_from_chunks(chunks: List[str], language: str = "english") -> List[str]:
    """
    Vereinheitlichte Satzsegmentierung auf Basis bereits vorverarbeiteter Chunks.
    - Läd punkt-Modelle bei Bedarf.
    - Entfernt Leerraum und leere Sätze.
    """
    # Zusammenführen der Chunks zu einem String (bewahrt Reihenfolge)
    text = "\n".join(c for c in chunks if c is not None)

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    sents = sent_tokenize(text, language=language)
    sents = [s.strip() for s in sents if s and s.strip()]
    return sents

def load_and_prepare_report(report_path: Path) -> List[str]:
    """Load, clean and chunk a single PDF report."""

    return chunk_text(clean_text(extract_text_from_pdfs([report_path])))


def load_and_prepare_reports(report_dir: Path) -> List[str]:
    """Load all PDF files from a directory and return prepared chunks."""

    pdf_paths = sorted(report_dir.glob("*.pdf"))
    raw_text = extract_text_from_pdfs(pdf_paths)
    return chunk_text(clean_text(raw_text))