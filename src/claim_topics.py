# src/claim_topics.py

from __future__ import annotations

from typing import Dict, List
from collections import Counter
import re

from src.claims_clustering import ClaimCluster


# Einfache, regelbasierte Zuordnung von Themen anhand von Keywords.
# Diese Liste kannst du später verfeinern/erweitern.
CLAIM_TOPIC_KEYWORDS: Dict[str, List[str]] = {
    # ---------------------------------------
    # KLIMA / EMISSIONEN / ENERGIE
    # ---------------------------------------
    "emissions": [

        "emission", "emissions", "co2", "co₂", "carbon", "carbon footprint",
        "carbon intensity", "ghg", "greenhouse gas", "greenhouse-gas",
        "scope 1", "scope1", "scope 2", "scope2", "scope 3", "scope3",
        "climate change", "global warming", "low-carbon",
    ],

    "energy": [

        "energy", "energy efficiency", "energy-efficient",
        "renewable", "renewables", "renewable energy",
        "solar", "photovoltaic", "pv", "wind", "onshore", "offshore",
        "hydro", "hydropower", "geothermal", "biomass",
        "green electricity", "green power",
    ],

    "transition": [

        "transition", "energy transition", "just transition",
        "transformation", "net zero pathway", "transition plan",
        "decarbonisation", "decarbonization", "net zero strategy",
        "coal phase out", "coal phase-out", "exit from coal",
        "phase-out of coal",
    ],

    # ---------------------------------------
    # SOZIALES / MENSCHEN / LIEFERKETTE
    # ---------------------------------------
    "social": [

        "social", "societal", "social responsibility",
        "community", "local community", "stakeholder engagement",
        "donations", "charitable", "volunteering", "pro bono",
    ],

    "human_rights_supply_chain": [

        "human rights", "labour rights", "labor rights",
        "child labour", "child labor", "forced labour", "forced labor",
        "modern slavery", "freedom of association",
        "collective bargaining", "living wage",
        "supply chain", "supply-chain", "value chain",
        "supplier", "suppliers", "sourcing", "responsible sourcing",
    ],

    "health_safety": [

        "health and safety", "occupational safety", "workplace safety",
        "occupational health", "lost time injury", "lost-time injury",
        "ltifr", "incident rate", "near miss", "near-miss",
        "wellbeing", "well-being",
    ],

    "diversity_inclusion": [

        "diversity", "equity", "inclusion", "dei",
        "gender diversity", "gender balance", "equal opportunities",
        "equal pay", "pay gap", "gender pay gap",
        "lgbt", "lgbti", "lgbtq", "lgbtq+",
    ],

    "human_capital": [

        "human capital", "training", "development", "talent development",
        "upskilling", "reskilling", "learning", "training hours",
        "employee engagement", "employee satisfaction",
        "turnover", "staff turnover", "employee retention",
    ],

    # ---------------------------------------
    # GOVERNANCE / ETHIK / COMPLIANCE
    # ---------------------------------------
    "governance": [

        "governance", "corporate governance", "board of directors",
        "supervisory board", "management board", "executive board",
        "remuneration", "compensation", "audit committee",
        "risk committee", "nomination committee",
    ],

    "ethics_compliance": [

        "compliance", "ethics", "code of conduct", "anti-corruption",
        "anti corruption", "anti-bribery", "anti bribery",
        "bribery", "corruption", "fraud", "money laundering",
        "sanctions", "antitrust", "competition law",
        "whistleblowing", "speak up", "integrity hotline",
    ],

    "data_privacy_cybersecurity": [

        "data protection", "data privacy", "gdpr", "personal data",
        "cybersecurity", "information security", "it security",
        "data breach", "cyber attack", "cyber-attack",
        "information security management", "iso 27001",
    ],

    "tax_transparency": [

        "tax", "tax transparency", "country-by-country reporting",
        "cbycr", "fair tax", "aggressive tax planning",
    ],

    # ---------------------------------------
    # UMWELT: ABFALL / WASSER / BIODIVERSITÄT / KREISLAUF
    # ---------------------------------------
    "waste_circularity": [

        "waste", "waste management", "hazardous waste", "non-hazardous waste",
        "landfill", "recycling", "recycled", "circular economy",
        "circularity", "resource efficiency", "material efficiency",
    ],

    "water": [

        "water", "water use", "water consumption", "water withdrawal",
        "water discharge", "wastewater", "effluent", "water scarcity",
        "water stress", "freshwater", "groundwater",
    ],

    "biodiversity": [

        "biodiversity", "ecosystem", "ecosystems", "habitat",
        "protected area", "protected areas", "nature conservation",
        "deforestation", "afforestation", "reforestation",
        "land use", "land-use change",
    ],

    "pollution_air_soil": [

        "air pollution", "soil pollution", "noise pollution",
        "particulate matter", "pm2.5", "pm10", "nox", "sox",
        "contamination", "spill", "spillage",
    ],

    # ---------------------------------------
    # PRODUKTE / KUNDEN / INNOVATION
    # ---------------------------------------
    "product_responsibility": [

        "product responsibility", "product safety", "product quality",
        "product stewardship", "eco-design", "ecodesign",
        "sustainable products", "green products",
        "life-cycle assessment", "lca",
    ],

    "customer": [

        "customer satisfaction", "customer loyalty", "customer privacy",
        "fair marketing", "responsible marketing", "customer service",
    ],

    "innovation_digitalization": [

        "innovation", "innovative", "research and development", "r&d",
        "digitalization", "digitalisation", "digital transformation",
        "smart", "internet of things", "iot", "artificial intelligence",
        "ai", "machine learning",
    ],

    # ---------------------------------------
    # COMMUNITY / ENGAGEMENT / SPENDEN
    # ---------------------------------------
    "community": [

        "local community", "community investment", "community projects",
        "philanthropy", "charitable giving", "donation", "donations",
        "volunteering", "employee volunteering",
    ],
}



def infer_cluster_topics(cluster: ClaimCluster, top_n: int = 3) -> List[str]:
    """
    Leitet aus den Sätzen eines Claim-Clusters Themen-Tags ab.
    Rückgabe ist eine sortierte Liste von Themen (höchste Trefferzahl zuerst).

    Vorgehen:
    - Alle Claim-Sätze in Kleinbuchstaben zusammenfügen
    - Keyword-Suche pro Thema
    - Themen mit >=1 Treffer zurückgeben (maximal top_n Themen)
    """
    text = " ".join([c.sentence.lower() for c in cluster.claims])
    topic_counts: Counter = Counter()

    for topic, keywords in CLAIM_TOPIC_KEYWORDS.items():
        for kw in keywords:
            # einfache Wort-/Phrasensuche, case-insensitive
            if re.search(r"\b" + re.escape(kw.lower()) + r"\b", text):
                topic_counts[topic] += 1

    if not topic_counts:
        return []

    # Themen nach Häufigkeit sortieren und auf top_n begrenzen
    topics_sorted = sorted(topic_counts.items(), key=lambda x: -x[1])
    topics = [t for t, _ in topics_sorted[:top_n]]
    return topics


def assign_topics_to_clusters(clusters: List[ClaimCluster], top_n: int = 3) -> None:
    """
    Mutiert die übergebenen ClaimCluster-Objekte in-place,
    indem die claim_topics-Felder gesetzt werden.
    """
    for cl in clusters:
        cl.claim_topics = infer_cluster_topics(cl, top_n=top_n)
