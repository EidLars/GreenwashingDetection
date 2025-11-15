# src/claims_extraction.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import sys

from tqdm import tqdm

# HF Transformers / Torch
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)
from transformers.pipelines.pt_utils import KeyDataset

# Zentrales Preprocessing (extern):
# Erwartet wird eine Funktion, die aus vorverarbeiteten Chunks Satz-Strings erzeugt.
# Siehe Anpassung in src/preprocessing.py unten.
from src.preprocessing import sentences_from_chunks


# ---------------------------------------------------------------------
# Öffentliche Datenstruktur (JSON-Schlüssel UNVERÄNDERT)
# ---------------------------------------------------------------------

@dataclass
class Claim:
    sentence: str
    p_yes: float
    p_no: float
    label: str  # "YES" oder "NO"


# ---------------------------------------------------------------------
# Interne Hilfsfunktionen: 1:1 aus funktionierender Datei übernommen
# (ohne PDF-/Tokenizer-Download-Outputs zu verändern)
# ---------------------------------------------------------------------

def _infer_labels(model: AutoModelForSequenceClassification) -> Tuple[str, str]:
    """
    Robust positive/negative Klasse aus der Modellkonfiguration ableiten.
    Bevorzugt semantische Bezeichnungen "yes"/"no"; Fallback: (LABEL_1, LABEL_0).
    """
    id2label = getattr(model.config, "id2label", None) or {}
    labels: List[str] = []
    if isinstance(id2label, dict) and id2label:
        try:
            norm = {int(k): str(v) for k, v in id2label.items()}
        except Exception:
            norm = {i: str(v) for i, v in enumerate(id2label.values())}
        labels = [norm[i].strip() for i in sorted(norm.keys())]

    if len(labels) == 2:
        lower = [s.lower() for s in labels]
        if "yes" in lower and "no" in lower:
            return labels[lower.index("yes")], labels[lower.index("no")]
        # Häufig: LABEL_0/1; zweite Klasse als positiv
        return labels[1], labels[0]

    # Generischer Fallback
    return "yes", "no"


def _build_pipeline(model_name: str, max_length: int) -> Tuple[TextClassificationPipeline, str, str]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    pos_label, neg_label = _infer_labels(model)
    device = 0 if torch.cuda.is_available() else -1
    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        max_length=max_length,
        task="text-classification",
    )
    return pipe, pos_label, neg_label


def _predict_probs(sentences: List[str], pipe: TextClassificationPipeline, pos_label: str, neg_label: str) -> List[Tuple[float, float]]:
    """
    Führt die Klassifikation durch und liefert (p_yes, p_no) pro Satz.
    Genau EIN tqdm-Fortschrittsbalken auf stdout.
    """
    if not sentences:
        return []

    p_list: List[Tuple[float, float]] = []
    dataset = [{"text": s} for s in sentences]

    for outputs in tqdm(
        pipe(
            KeyDataset(dataset, "text"),
            top_k=None,
            function_to_apply="softmax",
            truncation=True,
            padding=True,
        ),
        total=len(dataset),
        desc="Klassifiziere Sätze",
        file=sys.stdout,
        dynamic_ncols=True,
        leave=True,
    ):
        p_yes = 0.0
        p_no = 0.0
        if isinstance(outputs, list):
            for cand in outputs:
                label = str(cand.get("label", "")).strip()
                score = float(cand.get("score", 0.0))
                if label.lower() == pos_label.lower():
                    p_yes = score
                elif label.lower() == neg_label.lower():
                    p_no = score
        else:
            label = str(outputs.get("label", "")).strip()
            score = float(outputs.get("score", 0.0))
            if label.lower() == pos_label.lower():
                p_yes = score
                p_no = 1.0 - score
            elif label.lower() == neg_label.lower():
                p_no = score
                p_yes = 1.0 - score

        # Binär normalisieren (robust)
        if abs((p_yes + p_no) - 1.0) > 1e-3:
            p_no = max(0.0, min(1.0, 1.0 - p_yes))

        p_list.append((p_yes, p_no))
    return p_list


# ---------------------------------------------------------------------
# Öffentliche Extraktor-Klasse (Pipeline-API)
# ---------------------------------------------------------------------

class ClimateBERTClaimExtractor:
    """
    Wrapper, der die bestehende, validierte Logik kapselt und
    mit der Pipeline-Signatur kompatibel ist:
        extract(chunks: List[str]) -> List[Claim]
    """

    def __init__(
        self,
        model_name: str = "climatebert/environmental-claims",
        max_length: int = 512,
        threshold_yes: float = 0.5,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.threshold_yes = float(threshold_yes)
        self._pipe, self._pos_label, self._neg_label = _build_pipeline(model_name, max_length)

    def extract(self, chunks: List[str]) -> List[Claim]:
        """
        Erwartet vorverarbeitete Text-Chunks (externes Preprocessing).
        1) Satzsegmentierung über src.preprocessing.sentences_from_chunks(...)
        2) Klassifikation mit Softmax (p_yes, p_no)
        3) Label-Zuweisung: YES, wenn p_yes >= threshold_yes; sonst NO
        4) Rückgabe als List[Claim] (JSON-kompatibel via asdict)
        """
        # 1) Sätze (externes Preprocessing)
        sentences: List[str] = sentences_from_chunks(chunks, language="english")

        # 2) Klassifikation
        probs: List[Tuple[float, float]] = _predict_probs(sentences, self._pipe, self._pos_label, self._neg_label)

        # 3) Records -> Claims (JSON-Schlüssel unverändert)
        claims: List[Claim] = []
        pos_upper = self._pos_label.upper()
        neg_upper = self._neg_label.upper()
        for s, (p_yes, p_no) in zip(sentences, probs):
            label = pos_upper if p_yes >= self.threshold_yes else neg_upper
            if p_yes >= self.threshold_yes:
                claims.append(Claim(sentence=s, p_yes=float(p_yes), p_no=float(p_no), label=label))
        return claims
