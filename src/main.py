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
from src.kpi_extraction import FinancialKPI, extract_financial_kpis
from src.preprocessing import load_and_prepare_report
from src.claims_clustering import ClaimEmbedder, ClaimClusterer
from src.claim_topics import assign_topics_to_clusters
from src.claims_kpis_matching import save_batches_for_claim_file
from src.evaluation import analyse_matching_file


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
                    label=entry.get("label"),
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


def _extract_reporting_year_from_path(report_path: Path) -> str | None:
    """
    Extrahiert das Berichtsjahr aus dem Dateinamen der Ursprungs-PDF.
    Verwendet die hintersten 4 Ziffern im Stem (Dateiname ohne Suffix),
    z.B. 'bp_annual_report_2021' -> '2021'.

    Gibt None zurück, wenn kein 4-stelliger Jahresstring gefunden wird.
    """
    stem = report_path.stem
    m = re.search(r"(\d{4})(?!.*\d)", stem)  # letzte 4er-Zahl im Namen
    if m:
        return m.group(1)
    return None


def run_matching_batches(
    input_dir: Path | None = None,
    return_result: bool = False,
) -> Dict[str, object] | None:
    """
    Führt *nur* das Matching (Claims-Cluster + KPI-Subset) programmatisch aus,
    ohne einen LLM-Call zu machen.

    Es werden für alle vorhandenen Claims- und KPI-JSONs Matching-Dateien
    unter data/output/matching erzeugt. Pro Claims-File entsteht genau eine
    Datei mit der Struktur:

        {
          "company_key": ...,
          "claims_file": ...,
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

    Die Gruppierung der Reports erfolgt analog zur Evaluation über company_key.
    """

    chosen_input = (
        input_dir
        if input_dir is not None
        else Path(__file__).resolve().parents[1] / "data" / "input"
    )
    LOGGER.info("Starte MATCHING-PIPELINE für %s", chosen_input)

    output_root = Path(__file__).resolve().parents[1] / "data" / "output"
    claims_output_dir = output_root / "claims"
    kpi_output_dir = output_root / "kpis"
    matching_output_dir = output_root / "matching"
    matching_output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Claims und KPIs laden
    claims_per_report = _load_saved_claims(claims_output_dir)
    kpis_per_report = _load_saved_kpis(kpi_output_dir)

    if not claims_per_report:
        LOGGER.warning("Keine gespeicherten Claims gefunden unter %s", claims_output_dir)
    if not kpis_per_report:
        LOGGER.warning("Keine gespeicherten KPIs gefunden unter %s", kpi_output_dir)

    # Firmenpräfix-Logik (company_key)
    _TOKENS = [
        "annual", "sustainability", "csr", "esg",
        "report", "bericht", "nachhaltigkeitsbericht", "geschaeftsbericht",
        "integrated", "statement", "iar", "non-financial", "financial"
    ]
    _YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

    # Claims gruppieren
    claims_by_key: Dict[str, Dict[str, List[object]]] = {}
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

    # KPIs gruppieren
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

    # Embedder einmalig initialisieren und an save_batches_for_claim_file weitergeben
    embedder = ClaimEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    summary: Dict[str, object] = {
        "matching_files": {},
        "no_kpis_for_key": [],
        "no_batches_created": [],
    }

    # 2) Für jede company_key-Gruppe Matching-Dateien erzeugen
    for key, claim_files in claims_by_key.items():
        kpi_files = kpis_by_key.get(key, {})
        if not kpi_files:
            LOGGER.warning(
                "Keine KPI-Dateien zum Präfix '%s' gefunden – Matching für diese Gruppe übersprungen.",
                key,
            )
            summary["no_kpis_for_key"].append(key)
            continue

        all_kpis: List[FinancialKPI] = []
        for _kstem, klist in kpi_files.items():
            all_kpis.extend(klist)

        kpi_file_stems = list(kpi_files.keys())

        for c_stem, c_entries in claim_files.items():
            outfile = save_batches_for_claim_file(
                company_key=key,
                claims_file_stem=c_stem,
                c_entries=c_entries,
                all_kpis=all_kpis,
                kpi_file_stems=kpi_file_stems,
                output_dir=matching_output_dir,
                embedder=embedder,
                n_clusters=10,
                max_claims_per_batch=10,
                max_kpis_per_cluster=20,
            )
            if outfile is None:
                summary["no_batches_created"].append(c_stem)
            else:
                summary["matching_files"][c_stem] = str(outfile)

    if return_result:
        return summary

    # Standard: Kompakte Übersicht ausgeben
    print("\n=== MATCHING-ZUSAMMENFASSUNG ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return None


def run_pipeline(input_dir: Path, steps: Iterable[str] | None = None) -> Dict[str, object]:
    """Run the selected pipeline steps for the given input directory.

    Unterstützte Schritte:
      - "claims":     Claim-Extraktion aus Nachhaltigkeitsberichten
      - "kpis":       KPI-Extraktion aus Geschäftsberichten
      - "matching":   Matching & Batch-Bildung (Claims-Cluster + KPI-Subset → Matching-JSONs)
      - "evaluation": LLM-basierte Evaluation der Matching-Batches
      - "all":        alle obigen Schritte in der Reihenfolge
    """

    # -------------------------------------------------------------
    # 0) Schrittkonfiguration
    # -------------------------------------------------------------
    selected_steps: Set[str] = set(steps or {"claims", "kpis", "matching", "evaluation"})
    if "all" in selected_steps:
        selected_steps = {"claims", "kpis", "matching", "evaluation"}

    sustainability_dir = input_dir / "nachhaltigkeitsberichte"
    financial_dir = input_dir / "geschaeftsberichte"

    output_root = Path(__file__).resolve().parents[1] / "data" / "output"
    claims_output_dir = output_root / "claims"
    kpi_output_dir = output_root / "kpis"
    matching_output_dir = output_root / "matching"
    results_output_dir = output_root / "results"

    for folder in (claims_output_dir, kpi_output_dir, matching_output_dir, results_output_dir):
        folder.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1) Claim-Extraktion
    # -------------------------------------------------------------
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
                json.dumps(claim_output, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            LOGGER.info("Claims gespeichert unter %s", output_path)
    else:
        LOGGER.info("Überspringe Claim-Extraktion und lade bestehende Ergebnisse")
        claims_per_report = _load_saved_claims(claims_output_dir)

    # -------------------------------------------------------------
    # 2) KPI-Extraktion
    # -------------------------------------------------------------
    kpis_per_report: Dict[str, List[FinancialKPI]] = {}
    if "kpis" in selected_steps:
        LOGGER.info("Extrahiere finanzielle Kennzahlen mit dem LLM")
        for report_path in sorted(financial_dir.glob("*.pdf")):
            LOGGER.info("Verarbeite Geschäftsbericht %s", report_path.name)
            chunks = load_and_prepare_report(report_path)
            kpis = extract_financial_kpis(chunks)

            # Jahr aus Ursprungs-PDF bestimmen (hinterste 4 Ziffern im Dateinamen)
            reporting_year = _extract_reporting_year_from_path(report_path)
            if reporting_year is None:
                LOGGER.warning(
                    "Konnte kein Berichtsjahr aus dem Dateinamen %s extrahieren.",
                    report_path.name,
                )

            # Jahr an jede KPI hängen
            for kpi in kpis:
                kpi.reportingyear = reporting_year

            kpis_per_report[report_path.stem] = kpis

            # JSON-Ausgabe inkl. reportingyear
            kpi_output = [asdict(kpi) for kpi in kpis]
            output_path = kpi_output_dir / f"{report_path.stem}.json"
            output_path.write_text(
                json.dumps(kpi_output, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            LOGGER.info("KPIs gespeichert unter %s", output_path)
    else:
        LOGGER.info("Überspringe KPI-Extraktion und lade bestehende Ergebnisse")
        kpis_per_report = _load_saved_kpis(kpi_output_dir)

    # -------------------------------------------------------------
    # 3) Matching-Schritt (Claims-Cluster + KPI-Subset → Matching-JSONs)
    # -------------------------------------------------------------
    matching_summary: Dict[str, object] | None = None
    if "matching" in selected_steps or "evaluation" in selected_steps:
        LOGGER.info("Starte Matching-Schritt in der Pipeline")
        matching_summary = run_matching_batches(input_dir=input_dir, return_result=True)

    # -------------------------------------------------------------
    # 4) LLM-basierte Evaluation auf Basis der Matching-Dateien
    # -------------------------------------------------------------
    evaluations: Dict[str, Dict[str, object]] = {}

    if "evaluation" in selected_steps:
        LOGGER.info("Starte LLM-basierte Evaluation auf Basis der Matching-Dateien")

        for matching_path in sorted(matching_output_dir.glob("*.json")):
            try:
                matching_raw = matching_path.read_text(encoding="utf-8")
                matching_data = json.loads(matching_raw)
            except json.JSONDecodeError:
                LOGGER.warning("Konnte Matching-Datei %s nicht lesen – übersprungen", matching_path)
                continue

            company_key = matching_data.get("company_key")
            claims_file = matching_data.get("claims_file")
            kpi_files = matching_data.get("kpi_files", [])

            LOGGER.info(
                "Starte LLM-Evaluation für Matching-Datei %s (company_key=%s, claims_file=%s)",
                matching_path.name,
                company_key,
                claims_file,
            )

            # Batch für Batch mit dem LLM auswerten
            batch_results = analyse_matching_file(matching_path)

            # Evaluationsobjekt mit Header + Results
            evaluation_obj = {
                "company_key": company_key,
                "claims_file": claims_file,
                "kpi_files": kpi_files,
                "results": batch_results,  # Liste von {claim, kpi, relation, rationale}
            }

            out_stem = f"{matching_path.stem}__evaluated"
            evaluations[out_stem] = evaluation_obj

            out_path = results_output_dir / f"{out_stem}.json"
            out_path.write_text(
                json.dumps(evaluation_obj, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            LOGGER.info("Evaluation gespeichert unter %s", out_path)

        if not evaluations:
            LOGGER.warning(
                "Keine Evaluationsausgaben erzeugt – vermutlich fehlen Matching-Dateien unter %s",
                matching_output_dir,
            )
    else:
        LOGGER.info("Überspringe Evaluation und lade bestehende Ergebnisse")
        evaluations = _load_saved_evaluations(results_output_dir)

    # -------------------------------------------------------------
    # 5) Zusammenfassendes Pipeline-Resultat
    # -------------------------------------------------------------
    return {
        "claims": {stem: len(entries) for stem, entries in claims_per_report.items()},
        "kpis": {stem: len(entries) for stem, entries in kpis_per_report.items()},
        "matching": matching_summary,
        "evaluations": evaluations,
        "output_root": str(output_root),
    }


def parse_args() -> argparse.Namespace:
    """
    Parsed alle Kommandozeilenparameter für die Greenwashing-Pipeline.

    Verfügbare Modi:
      --steps <claims|kpis|matching|evaluation|all>
        Generische Steuerung der Pipeline-Schritte. Mehrere Werte sind möglich,
        z.B.:
          --steps claims kpis
          --steps kpis matching evaluation
          --steps all   (Voreinstellung, entspricht der vollständigen Pipeline)
    """

    parser = argparse.ArgumentParser(description=__doc__)

    # Eingabeverzeichnis
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "input",
        help="Pfad zum Eingabeverzeichnis mit PDF-Berichten.",
    )

    # Generisches Steps-Interface
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["claims", "kpis", "matching", "evaluation", "all"],
        default=["all"],
        help=(
            "Auszuführende Pipeline-Schritte. "
            "Standard: 'all' für die vollständige Pipeline."
        ),
    )

    # Optionaler Ausgabeort
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optionaler Pfad zur Speicherung der Pipeline-Ausgabe im JSON-Format.",
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
    """
    chosen_input = (
        input_dir
        if input_dir is not None
        else Path(__file__).resolve().parents[1] / "data" / "input"
    )
    LOGGER.info("Starte QUICK TEST: nur Evaluation (ohne CLI) für %s", chosen_input)
    result = run_pipeline(chosen_input, steps=["evaluation"])

    if return_result:
        return result

    print(json.dumps(result, ensure_ascii=False, indent=2))


def quick_test_claims_clustering(
    claims_output_dir: Path | None = None,
    n_clusters: int = 10,
    return_result: bool = False,
) -> Dict[str, object] | None:
    """
    Führt *nur* das Claim-Clustering (inkl. Topic-Zuordnung) programmatisch aus.
    """

    if claims_output_dir is None:
        claims_output_dir = (
            Path(__file__).resolve().parents[1] / "data" / "output" / "claims"
        )

    LOGGER.info(
        "Starte QUICK TEST: Claim-Clustering für gespeicherte Claims in %s",
        claims_output_dir,
    )

    # 1) Claims aus JSON als Claim-Dataclasses laden
    claims_per_report: Dict[str, List[Claim]] = {}
    for json_path in sorted(claims_output_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("Konnte Claims-Datei %s nicht lesen – übersprungen", json_path)
            continue

        entries: List[Claim] = []
        for entry in data:
            try:
                entries.append(Claim(**entry))
            except TypeError as exc:
                LOGGER.warning(
                    "Ungültiger Claim-Eintrag in %s: %s (Fehler: %s)",
                    json_path,
                    entry,
                    exc,
                )
        if not entries:
            LOGGER.warning("Keine gültigen Claims in %s gefunden – übersprungen", json_path)
            continue

        claims_per_report[json_path.stem] = entries

    if not claims_per_report:
        LOGGER.warning("Keine Claims-JSONs gefunden – Abbruch.")
        return None

    # 2) Embedder einmal initialisieren
    embedder = ClaimEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    results: Dict[str, object] = {}

    # 3) Pro Report: Claims clustern und Topics zuweisen
    for stem, claims in claims_per_report.items():
        LOGGER.info("Clustere Claims für Report '%s' (%d Claims)", stem, len(claims))

        eff_n_clusters = min(n_clusters, len(claims))
        if eff_n_clusters <= 0:
            LOGGER.warning("Report '%s' hat keine Claims – übersprungen.", stem)
            continue

        clusterer = ClaimClusterer(
            embedder=embedder,
            n_clusters=eff_n_clusters,
            random_state=42,
        )
        clusters = clusterer.cluster_claims(claims)

        assign_topics_to_clusters(clusters, top_n=3)

        print("=" * 120)
        print(f"REPORT: {stem}")
        print(f"Anzahl Claims:   {len(claims)}")
        print(f"Anzahl Cluster:  {len(clusters)}")
        print("-" * 120)

        for cl in clusters:
            print(
                f"Cluster {cl.cluster_id}: "
                f"{len(cl.claims)} Claims, Topics: {cl.claim_topics}"
            )
            for c in cl.claims[:3]:
                print(f"  - {c.sentence}")
            print("-" * 80)

        results[stem] = {
            "num_claims": len(claims),
            "num_clusters": len(clusters),
            "clusters": [
                {
                    "cluster_id": cl.cluster_id,
                    "num_claims": len(cl.claims),
                    "topics": cl.claim_topics,
                }
                for cl in clusters
            ],
        }

    if return_result:
        return results

    print("\n\n=== Zusammenfassung Claim-Clustering (pro Report) ===")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return None


def quick_test_matching(
    input_dir: Path | None = None,
    return_result: bool = False,
) -> Dict[str, object] | None:
    """
    Führt *nur* das Matching (Claims-Cluster + KPI-Subset) programmatisch aus,
    ohne CLI und ohne LLM-Call.
    """
    return run_matching_batches(input_dir=input_dir, return_result=return_result)

def random_claims_for_evaluation():
    """
            Lädt einen festen PDF-Bericht, extrahiert Chunks,
            klassifiziert alle Sätze (YES/NO) und gibt 10 zufällige Claims aus.
            """
    from src.preprocessing import load_and_prepare_report
    from src.claims_extraction import sample_yes_no_claims_from_chunks

    # Fester Pfad innerhalb deines Projekts
    pdf_path = Path(__file__).resolve().parents[
                   1] / "data" / "input" / "nachhaltigkeitsberichte" / "nestle-non-financial-statement-2024.pdf"

    print(f"Nutze PDF: {pdf_path}")

    # Text vorbereiten
    chunks = load_and_prepare_report(pdf_path)

    # YES+NO Claims, 10 zufällig
    claims = sample_yes_no_claims_from_chunks(chunks, n_samples=10)

def main() -> None:
    args = parse_args()
    steps = args.steps

    result = run_pipeline(args.input_dir, steps=steps)

    if args.output:
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))



if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
