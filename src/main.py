"""Command line entry point for the greenwashing detection prototype."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Set
import re

from src.claims_extraction import Claim, ClimateBERTClaimExtractor
from src.evaluation import analyse_greenwashing
from src.kpi_extraction import FinancialKPI, extract_financial_kpis
from src.preprocessing import load_and_prepare_report

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def _load_saved_claims(claims_output_dir: Path) -> Dict[str, List[object]]:
    """
    Lädt Claims im Format:
    [
      {"sentence": "...", "p_yes": float, "p_no": float, "label": "yes" },
      ...
    ]

    Rückgabe: { report_name: [Objekt(text, p_yes, p_no, label), ...] }
    """

    claims_per_report: Dict[str, List[object]] = {}

    for json_path in sorted(claims_output_dir.glob("*.json")):
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            LOGGER.warning("Kann Claims-Datei %s nicht lesen – übersprungen", json_path)
            continue

        converted = []
        for entry in raw:
            if "sentence" not in entry:
                LOGGER.warning("Eintrag ohne 'sentence' in %s – übersprungen", json_path)
                continue

            # Wir erzeugen ein einfaches Objekt mit den benötigten Feldern
            converted.append(
                SimpleNamespace(
                    text=entry["sentence"].strip(),
                    p_yes=entry.get("p_yes"),
                    p_no=entry.get("p_no"),
                    label=entry.get("label")
                )
            )

        claims_per_report[json_path.stem] = converted

    return claims_per_report


def _load_saved_kpis(kpi_output_dir: Path) -> Dict[str, List[FinancialKPI]]:
    """Load stored KPI results to support isolated evaluations."""

    kpis_per_report: Dict[str, List[FinancialKPI]] = {}
    for json_path in sorted(kpi_output_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("Konnte KPI-Datei %s nicht lesen", json_path)
            continue
        entries: List[FinancialKPI] = []
        for entry in data:
            try:
                entries.append(FinancialKPI(**entry))
            except TypeError:
                LOGGER.warning("Ungültiger KPI in %s: %s", json_path, entry)
        kpis_per_report[json_path.stem] = entries
    return kpis_per_report


def _load_saved_evaluations(results_output_dir: Path) -> Dict[str, Dict[str, object]]:
    """Return previously stored evaluation outputs."""

    evaluations: Dict[str, Dict[str, object]] = {}
    for json_path in sorted(results_output_dir.glob("*.json")):
        try:
            evaluations[json_path.stem] = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("Konnte Evaluation %s nicht lesen", json_path)
    return evaluations


def run_pipeline(input_dir: Path, steps: Iterable[str] | None = None) -> Dict[str, object]:
    """Run the selected pipeline steps for the given input directory."""

    selected_steps: Set[str] = set(steps or {"claims", "kpis", "evaluation"})
    if "all" in selected_steps:
        selected_steps = {"claims", "kpis", "evaluation"}

    sustainability_dir = input_dir / "nachhaltigkeitsberichte"
    financial_dir = input_dir / "geschaeftsberichte"

    output_root = Path(__file__).resolve().parents[1] / "data" / "output"
    claims_output_dir = output_root / "claims"
    kpi_output_dir = output_root / "kps"
    results_output_dir = output_root / "results"
    for folder in (claims_output_dir, kpi_output_dir, results_output_dir):
        folder.mkdir(parents=True, exist_ok=True)

    claims_per_report: Dict[str, List[Claim]] = {}
    if "claims" in selected_steps:
        LOGGER.info("Extrahiere Claims mit ClimateBERT")
        extractor = ClimateBERTClaimExtractor()
        for report_path in sorted(sustainability_dir.glob("*.pdf")):
            LOGGER.info("Verarbeite Nachhaltigkeitsbericht %s", report_path.name)
            chunks = load_and_prepare_report(report_path)
            claims = extractor.extract(chunks)
            claims_per_report[report_path.stem] = claims
            claim_output = [asdict(claim) for claim in claims]
            output_path = claims_output_dir / f"{report_path.stem}.json"
            output_path.write_text(
                json.dumps(claim_output, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            LOGGER.info("Claims gespeichert unter %s", output_path)
    else:
        LOGGER.info("Überspringe Claim-Extraktion und lade bestehende Ergebnisse")
        claims_per_report = _load_saved_claims(claims_output_dir)

    kpis_per_report: Dict[str, List[FinancialKPI]] = {}
    if "kpis" in selected_steps:
        LOGGER.info("Extrahiere finanzielle Kennzahlen mit dem LLM")
        for report_path in sorted(financial_dir.glob("*.pdf")):
            LOGGER.info("Verarbeite Geschäftsbericht %s", report_path.name)
            chunks = load_and_prepare_report(report_path)
            kpis = extract_financial_kpis(chunks)
            kpis_per_report[report_path.stem] = kpis
            kpi_output = [asdict(kpi) for kpi in kpis]
            output_path = kpi_output_dir / f"{report_path.stem}.json"
            output_path.write_text(
                json.dumps(kpi_output, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            LOGGER.info("KPIs gespeichert unter %s", output_path)
    else:
        LOGGER.info("Überspringe KPI-Extraktion und lade bestehende Ergebnisse")
        kpis_per_report = _load_saved_kpis(kpi_output_dir)

    evaluations: Dict[str, Dict[str, object]] = {}

    if "evaluation" in selected_steps:
        LOGGER.info("Bewerte Konsistenz zwischen Claims und Kennzahlen")

        # Schlüsselwörter/TRIGGER, ab denen der Firmenpräfix endet
        _TOKENS = [
            "annual", "sustainability", "csr", "esg",
            "report", "bericht", "nachhaltigkeitsbericht", "geschaeftsbericht",
            "integrated", "statement", "iar"
        ]
        _YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

        # 1) Claims nach einfachem Präfix gruppieren (inline-Präfixableitung, keine Hilfsfunktion)
        claims_by_key: Dict[str, Dict[str, List[Claim]]] = {}
        for c_stem, c_entries in claims_per_report.items():
            s = c_stem.lower()
            earliest = len(s)
            for tok in _TOKENS:
                idx = s.find(tok)
                if idx != -1 and idx < earliest:
                    earliest = idx
            m = _YEAR_RE.search(s)
            if m and m.start() < earliest:
                earliest = m.start()
            key = s[:earliest].strip() or s.strip()
            key = re.sub(r"\s+", " ", key)
            claims_by_key.setdefault(key, {})[c_stem] = c_entries

        # 2) KPIs nach demselben Präfix gruppieren (inline, identische Logik)
        kpis_by_key: Dict[str, Dict[str, List[FinancialKPI]]] = {}
        for k_stem, k_entries in kpis_per_report.items():
            s = k_stem.lower()
            earliest = len(s)
            for tok in _TOKENS:
                idx = s.find(tok)
                if idx != -1 and idx < earliest:
                    earliest = idx
            m = _YEAR_RE.search(s)
            if m and m.start() < earliest:
                earliest = m.start()
            key = s[:earliest].strip() or s.strip()
            key = re.sub(r"\s+", " ", key)
            kpis_by_key.setdefault(key, {})[k_stem] = k_entries

        evaluations: Dict[str, Dict[str, object]] = {}
        matched = 0

        # 3) Für jedes Claims-File: alle KPI-Dateien mit gleichem Präfix vereinigen und evaluieren
        for key, claim_files in claims_by_key.items():
            kpi_files = kpis_by_key.get(key, {})
            if not kpi_files:
                LOGGER.warning("Keine KPI-Dateien zum Präfix '%s' gefunden – überspringe.", key)
                continue

            all_kpis: List[FinancialKPI] = []
            for _kstem, klist in kpi_files.items():
                all_kpis.extend(klist)

            for c_stem, c_entries in claim_files.items():
                evaluation = analyse_greenwashing(
                    [claim.text for claim in c_entries],
                    all_kpis,
                )
                evaluation["anzahl_claims"] = len(c_entries)
                evaluation["anzahl_kpis"] = len(all_kpis)
                evaluation["company_key"] = key
                evaluation["paired_claims_file"] = c_stem
                evaluation["paired_kpi_files"] = sorted(list(kpi_files.keys()))

                out_stem = f"{c_stem}__vs__ALL_KPIS__{key}"
                evaluations[out_stem] = evaluation

                output_path = results_output_dir / f"{out_stem}.json"
                output_path.write_text(
                    json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                LOGGER.info("Evaluation gespeichert unter %s", output_path)
                matched += 1

        if matched == 0:
            LOGGER.warning("Keine passenden Claim–KPI-Paare anhand des Präfixes gefunden – Evaluation übersprungen")
    else:
        LOGGER.info("Überspringe Evaluation und lade bestehende Ergebnisse")
        evaluations = _load_saved_evaluations(results_output_dir)

    return {
        "claims": {stem: len(entries) for stem, entries in claims_per_report.items()},
        "kpis": {stem: len(entries) for stem, entries in kpis_per_report.items()},
        "evaluations": evaluations,
        "output_root": str(output_root),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "input",
        help="Ordner mit den PDF-Berichten",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["claims", "kpis", "evaluation", "all"],
        default=["all"],
        help=(
            "Welche Teile der Pipeline ausgeführt werden sollen. "
            "Standard ist 'all' für die komplette Verarbeitung."
        ),
    )
    parser.add_argument(
        "--only-claims",
        action="store_true",
        help="Nur die Claim-Extraktion ausführen (überschreibt --steps).",
    )
    parser.add_argument(
        "--only-kpis",
        action="store_true",
        help="Nur die KPI-Extraktion ausführen (überschreibt --steps).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optionaler Pfad für die JSON-Ausgabe",
    )
    return parser.parse_args()

def quick_test_claims(
    input_dir: Path | None = None,
    return_result: bool = False,
) -> Dict[str, object] | None:
    """
    Führt *nur* die Claim-Extraktion programmatisch aus, ohne CLI.
    Beispielaufruf (IDE/Notebook):
        from path.to.cli_module import quick_test_claims
        quick_test_claims()
    """
    chosen_input = (
        input_dir
        if input_dir is not None
        else Path(__file__).resolve().parents[1] / "data" / "input"
    )
    LOGGER.info("Starte QUICK TEST: nur Claims (ohne CLI) für %s", chosen_input)
    result = run_pipeline(chosen_input, steps=["claims"])
    if return_result:
        return result
    # Standard: kompaktes Pretty-Print zur unmittelbaren Sichtkontrolle
    print(json.dumps(result, ensure_ascii=False, indent=2))


def quick_test_kpis(
    input_dir: Path | None = None,
    return_result: bool = False,
) -> Dict[str, object] | None:
    """
    Führt *nur* die KPI-Extraktion programmatisch aus, ohne CLI.
    Beispielaufruf (IDE/Notebook):
        from path.to.cli_module import quick_test_kpis
        quick_test_kpis()
    """
    chosen_input = (
        input_dir
        if input_dir is not None
        else Path(__file__).resolve().parents[1] / "data" / "input"
    )
    LOGGER.info("Starte QUICK TEST: nur KPIs (ohne CLI) für %s", chosen_input)
    result = run_pipeline(chosen_input, steps=["kpis"])
    if return_result:
        return result
    print(json.dumps(result, ensure_ascii=False, indent=2))

def quick_test_evaluation(
    input_dir: Path | None = None,
    return_result: bool = False,
) -> Dict[str, object] | None:
    """
    Führt *nur* die Evaluation programmatisch aus, ohne CLI.
    Erwartet, dass Claims und KPIs bereits als JSON unter data/output/claims und
    data/output/kpis vorliegen (siehe _load_saved_claims / _load_saved_kpis).

    Beispielaufruf (IDE/Notebook):
        from path.to.cli_module import quick_test_evaluation
        quick_test_evaluation()
    """
    chosen_input = (
        input_dir
        if input_dir is not None
        else Path(__file__).resolve().parents[1] / "data" / "input"
    )
    LOGGER.info("Starte QUICK TEST: nur Evaluation (ohne CLI) für %s", chosen_input)

    # Wichtig: Nur 'evaluation' ausführen – Claims/KPIs werden aus JSON geladen.
    result = run_pipeline(chosen_input, steps=["evaluation"])

    if return_result:
        return result

    # Kompaktes Pretty-Print zur unmittelbaren Sichtkontrolle
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:  # pragma: no cover - thin CLI wrapper
    args = parse_args()
    if args.only_claims and args.only_kpis:
        steps = ["claims", "kpis"]
    elif args.only_claims:
        steps = ["claims"]
    elif args.only_kpis:
        steps = ["kpis"]
    else:
        steps = args.steps or ["all"]
    result = run_pipeline(args.input_dir, steps=steps)
    json_output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(json_output, encoding="utf-8")
        LOGGER.info("Ergebnis gespeichert unter %s", args.output)
    else:
        print(json_output)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    quick_test_claims()