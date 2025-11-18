CLAIM_TO_KPI_CATEGORY_MAP = {
    # ---------------------------------------------------------------
    # KLIMA / ENERGIE / TRANSFORMATION
    # ---------------------------------------------------------------
    "emissions": [
        "emissions_scope1",
        "emissions_scope2",
        "emissions_scope3",
        "emission_intensity",
    ],
    "energy": [
        "energy_consumption",
        "emission_intensity",
        "operational_costs",   # energiebezogene Kosten
        "capex_opex",          # energiebezogene Investitionen
    ],
    "transition": [
        "capex_opex",          # Transformations-/Transition-Investitionen
        "revenue_profit",      # Verschiebung der Geschäftssegmente im Umsatz/Gewinn
        "cashflow",            # Free Cashflow / Operating Cashflow unter Transition-Strategie
    ],

    # ---------------------------------------------------------------
    # SOZIALES / MENSCHEN / LIEFERKETTE
    # ---------------------------------------------------------------
    "social": [
        "operational_costs",   # z.B. Personalaufwand, Sozialprogramme
    ],
    "human_rights_supply_chain": [
        "operational_costs",   # Audit-, Monitoring-, Compliance-Kosten in Lieferketten
    ],
    "health_safety": [
        "operational_costs",   # H&S-Kosten, Schutzmaßnahmen, Trainings
    ],
    "diversity_inclusion": [
        "operational_costs",   # Programme, Trainings, DEI-Initiativen
    ],
    "human_capital": [
        "operational_costs",   # Personal-, Schulungs- und Entwicklungskosten
    ],

    # ---------------------------------------------------------------
    # GOVERNANCE / ETHIK / COMPLIANCE / STEUERN
    # ---------------------------------------------------------------
    "governance": [
        "capital_structure",   # Verschuldung, Eigenkapital etc.
        "revenue_profit",      # Governance-Effekte auf Profitabilität
        "cashflow",            # Governance-Effekte auf Cash Generierung
    ],
    "ethics_compliance": [
        "operational_costs",   # Compliance-Kosten
        "revenue_profit",      # potenziell Bußgelder, Strafen im Ergebnis
    ],
    "data_privacy_cybersecurity": [
        "capex_opex",          # IT-/Cybersecurity-Investitionen
        "operational_costs",   # laufende Sicherheits-/Datenschutzkosten
    ],
    "tax_transparency": [
        "revenue_profit",      # Steueraufwand im Gewinn- und Verlustkonto
        "cashflow",            # Steuerzahlungen im Cashflow
    ],

    # ---------------------------------------------------------------
    # UMWELT: ABFALL / WASSER / BIODIVERSITÄT / VERSCHMUTZUNG
    # ---------------------------------------------------------------
    "waste_circularity": [
        "operational_costs",   # Entsorgungs-, Recycling-, Kreislaufkosten
        "capex_opex",          # Investitionen in Kreislauf-/Recyclinganlagen
    ],
    "water": [
        "operational_costs",   # Wasserbezogene Betriebskosten, Aufbereitung
        "capex_opex",          # Investitionen in Wasserinfrastruktur
    ],
    "biodiversity": [
        "capex_opex",          # Biodiversitätsprogramme, Renaturierungsinvestitionen
    ],
    "pollution_air_soil": [
        "operational_costs",   # Luft-/Boden-/Lärmschutzmaßnahmen
        "capex_opex",          # Investitionen in Filter, Anlagen, Schutztechnik
    ],

    # ---------------------------------------------------------------
    # PRODUKTE / KUNDEN / INNOVATION
    # ---------------------------------------------------------------
    "product_responsibility": [
        "operational_costs",   # Qualitäts-, Sicherheits-, Rückrufkosten
        "revenue_profit",      # Auswirkungen auf Umsatz und Marge (z.B. durch Rückrufe)
    ],
    "customer": [
        "revenue_profit",      # Umsatz, Wiederholungskäufe, Kundenzufriedenheit
        "cashflow",            # Zahlungsströme aus Kundenbeziehungen
    ],
    "innovation_digitalization": [
        "capex_opex",          # F&E-, Digitalisierungs-, Technologieinvestitionen
        "operational_costs",   # laufende digitale Betriebs-/Lizenzkosten
    ],

    # ---------------------------------------------------------------
    # COMMUNITY / ENGAGEMENT / SPENDEN
    # ---------------------------------------------------------------
    "community": [
        "operational_costs",   # Spenden, Sponsoring, Community-Programme
        "cashflow",            # ggf. Mittelabflüsse aus Community-Investitionen
    ],
}
