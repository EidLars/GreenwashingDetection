"""Matching der Claims mit den KPIs"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union

from src.claims_extraction import Claim
from src.kpi_extraction import FinancialKPI
from src.claims_clustering import ClaimEmbedder, ClaimClusterer, ClaimCluster
from src.claim_topics import assign_topics_to_clusters
from src.config_mapping import CLAIM_TO_KPI_CATEGORY_MAP

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hilfsfunktion: generische Claim-Objekte in Claim-Dataclasses überführen
# ---------------------------------------------------------------------------

def convert_to_claims(c_entries: Iterable[Union[Claim, object]]) -> List[Claim]:
    """
    Konvertiert eine Liste von beliebigen Claim-Einträgen (z.B. Claim-Dataclasses
    oder SimpleNamespace mit Attributen text/p_yes/p_no/label) in eine
    einheitliche List[Claim].

    Erwartete Felder bei Nicht-Claim-Objekten:
        - text (str)
        - p_yes (optional, float)
        - p_no (optional, float)
        - label (optional, str)
    """
    claims: List[Claim] = []

    for c in c_entries:
        if isinstance(c, Claim):
            # Bereits ein Claim-Objekt
            claims.append(c)
            continue

        text = getattr(c, "text", "") or ""
        if not text.strip():
            continue

        p_yes = getattr(c, "p_yes", 1.0)
        if p_yes is None:
            p_yes = 1.0

        p_no = getattr(c, "p_no", 0.0)
        if p_no is None:
            p_no = 0.0

        label = getattr(c, "label", "YES") or "YES"

        claims.append(
            Claim(
                sentence=text.strip(),
                p_yes=float(p_yes),
                p_no=float(p_no),
                label=str(label),
            )
        )

    return claims


# ---------------------------------------------------------------------------
# Matching: Claim-Cluster mit KPI-Subset
# ---------------------------------------------------------------------------

def match_kpis_to_cluster(
    cluster: ClaimCluster,
    all_kpis: List[FinancialKPI],
    max_kpis: int = 20,
) -> List[FinancialKPI]:
    """
    Wählt für einen Claim-Cluster ein relevantes KPI-Subset auf Basis
    der Claim-Topics und der KPI-Kategorien aus.

    Voraussetzungen:
    - ClaimCluster.claim_topics: Liste von Topics (z.B. ["emissions", "energy"])
    - FinancialKPI.kategorie: String, der eine der vordefinierten KPI-Kategorien
      repräsentiert (z.B. "emissions_scope1", "revenue_profit" usw.).
    """
    allowed_categories: Set[str] = set()
    for topic in getattr(cluster, "claim_topics", []) or []:
        allowed_categories.update(CLAIM_TO_KPI_CATEGORY_MAP.get(topic, []))

    if not allowed_categories:
        return []

    candidates = [
        k for k in all_kpis
        if getattr(k, "kategorie", None) in allowed_categories
    ]

    # Priorisierung: Emissions-/Energiekategorien vor anderen
    def _priority(k: FinancialKPI) -> int:
        cat = getattr(k, "kategorie", "") or ""
        if cat.startswith("emissions_") or cat == "emission_intensity":
            return 0
        if cat in {"energy_consumption", "capex_opex"}:
            return 1
        if cat in {"revenue_profit", "cashflow"}:
            return 2
        return 3

    candidates.sort(key=_priority)
    return candidates[:max_kpis]


# ---------------------------------------------------------------------------
# Batches für EIN Claims-File eines Unternehmens aufbauen
# ---------------------------------------------------------------------------

def build_batches_for_claim_file(
    company_key: str,
    claims_file_stem: str,
    c_entries: Iterable[Union[Claim, object]],
    all_kpis: List[FinancialKPI],
    kpi_file_stems: Iterable[str],
    embedder: ClaimEmbedder,
    n_clusters: int = 10,
    max_claims_per_batch: int = 10,
    max_kpis_per_cluster: int = 20,
) -> Optional[Dict[str, Any]]:
    """
    Erzeugt für ein Claims-File (einen Nachhaltigkeitsbericht) alle
    Cluster- und KPI-basierten Batches in der Form:

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
                "meta": {
                    "cluster_id": ...,
                    "num_claims": ...,
                    "num_kpis": ...
                }
            },
            ...
        ]
    }

    Die batch_id zählt innerhalb dieses Reports von 0 aufwärts.
    """
    # Schritt 1: Claims normalisieren
    claims = convert_to_claims(c_entries)
    if not claims:
        LOGGER.warning(
            "Keine gültigen Claims für %s (company_key=%s) – keine Batches erzeugt.",
            claims_file_stem,
            company_key,
        )
        return None

    # Schritt 2: Clustering + Topics
    eff_n_clusters = min(n_clusters, len(claims))
    if eff_n_clusters <= 0:
        LOGGER.warning(
            "Report '%s' hat keine Claims nach Konvertierung – keine Batches erzeugt.",
            claims_file_stem,
        )
        return None

    clusterer = ClaimClusterer(
        embedder=embedder,
        n_clusters=eff_n_clusters,
        random_state=42,
    )
    clusters: List[ClaimCluster] = clusterer.cluster_claims(claims)
    assign_topics_to_clusters(clusters, top_n=3)

    # Schritt 3: Batches über alle Cluster
    batches: List[Dict[str, Any]] = []
    batch_counter = 0

    for cluster in clusters:
        matched_kpis = match_kpis_to_cluster(
            cluster=cluster,
            all_kpis=all_kpis,
            max_kpis=max_kpis_per_cluster,
        )
        if not matched_kpis:
            # Cluster ohne passende KPIs können übersprungen werden
            continue

        # Claims in diesem Cluster
        cluster_claims: List[Dict[str, Any]] = []
        for i, c in enumerate(cluster.claims):
            cluster_claims.append(
                {
                    "claim_id": f"{claims_file_stem}__cluster_{cluster.cluster_id}__C_{i}",
                    "text": c.sentence,
                    "p_yes": c.p_yes,
                    "label": c.label,
                }
            )

        # KPI-Records (ohne globale ID, nur inhaltlich)
        kpi_records: List[Dict[str, Any]] = [asdict(k) for k in matched_kpis]

        # Duplikate innerhalb des KPI-Sets entfernen
        kpi_records = deduplicate_kpi_records(kpi_records)

        # Claims in Batches aufteilen
        for start in range(0, len(cluster_claims), max_claims_per_batch):
            claims_batch = cluster_claims[start: start + max_claims_per_batch]

            batch = {
                "batch_id": batch_counter,
                "topics": getattr(cluster, "claim_topics", []) or [],
                "claims": claims_batch,
                "kpis": kpi_records,
                "meta": {
                    "cluster_id": cluster.cluster_id,
                    "num_claims": len(claims_batch),
                    "num_kpis": len(kpi_records),  # aktualisierte Anzahl nach Deduplikation
                },
            }
            batches.append(batch)
            batch_counter += 1

    if not batches:
        LOGGER.warning(
            "Für %s (company_key=%s) konnten keine Batches erzeugt werden.",
            claims_file_stem,
            company_key,
        )
        return None

    payload: Dict[str, Any] = {
        "company_key": company_key,
        "claims_file": claims_file_stem,
        "kpi_files": sorted(kpi_file_stems),
        "batches": batches,
    }
    return payload


# ---------------------------------------------------------------------------
# Batches speichern: EINE JSON-Datei pro Claims-Report
# ---------------------------------------------------------------------------

def save_batches_for_claim_file(
    company_key: str,
    claims_file_stem: str,
    c_entries: Iterable[Union[Claim, object]],
    all_kpis: List[FinancialKPI],
    kpi_file_stems: Iterable[str],
    output_dir: Path,
    embedder: Optional[ClaimEmbedder] = None,
    n_clusters: int = 10,
    max_claims_per_batch: int = 10,
    max_kpis_per_cluster: int = 20,
) -> Optional[Path]:
    """
    High-Level-Funktion:
    - baut alle Batches für ein Claims-File,
    - speichert sie als eine JSON-Datei unter output_dir,
    - gibt den Pfad zur Datei zurück (oder None, falls nichts erzeugt wurde).

    Dateiname:
        <claims_file_stem>__matching__<company_key>.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Embedder optional von außen wiederverwenden (Performance)
    if embedder is None:
        embedder = ClaimEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    payload = build_batches_for_claim_file(
        company_key=company_key,
        claims_file_stem=claims_file_stem,
        c_entries=c_entries,
        all_kpis=all_kpis,
        kpi_file_stems=kpi_file_stems,
        embedder=embedder,
        n_clusters=n_clusters,
        max_claims_per_batch=max_claims_per_batch,
        max_kpis_per_cluster=max_kpis_per_cluster,
    )

    if payload is None:
        return None

    out_stem = f"{claims_file_stem}__matching__{company_key}"
    outfile = output_dir / f"{out_stem}.json"
    outfile.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    LOGGER.info(
        "Matching-Batches für Claims-File %s (company_key=%s) gespeichert unter %s",
        claims_file_stem,
        company_key,
        outfile,
    )
    return outfile

# ---------------------------------------------------------------------------
# Doppelte KPIs in den Batches löschen
# ---------------------------------------------------------------------------
def deduplicate_kpi_records(kpi_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Entfernt doppelte KPI-Einträge innerhalb eines Batches.

    Zwei KPIs gelten als Duplikat, wenn sie in den folgenden Feldern identisch sind:
    - kpi (Bezeichnung)
    - value / wert
    - unit / einheit
    - kategorie
    - reportingyear

    Unterschiedliche Werte (z.B. verschiedene Personen, Jahre, Beträge)
    werden bewusst NICHT zusammengefasst.
    """
    seen: Set[tuple] = set()
    unique: List[Dict[str, Any]] = []

    for rec in kpi_records:
        # Feldnamen robust behandeln (value/wert, unit/einheit)
        name = rec.get("kpi")
        value = rec.get("value", rec.get("wert"))
        unit = rec.get("unit", rec.get("einheit"))
        cat = rec.get("kategorie")
        year = rec.get("reportingyear")

        key = (name, value, unit, cat, year)
        if key in seen:
            continue
        seen.add(key)
        unique.append(rec)

    return unique
