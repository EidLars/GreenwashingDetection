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
        "operational_costs",
        "capex_opex",
    ],
    "transition": [
        "capex_opex",
        "revenue_profit",
        "cashflow",
    ],

    # ---------------------------------------------------------------
    # SOZIALES / MENSCHEN / LIEFERKETTE
    # ---------------------------------------------------------------
    "social": [
        "operational_costs",
    ],
    "human_rights_supply_chain": [
        "operational_costs",
    ],
    "health_safety": [
        "operational_costs",
    ],
    "diversity_inclusion": [
        "operational_costs",
    ],
    "human_capital": [
        "operational_costs",
    ],

    # ---------------------------------------------------------------
    # GOVERNANCE / ETHIK / COMPLIANCE / STEUERN
    # ---------------------------------------------------------------
    "governance": [
        "capital_structure",
        "revenue_profit",
        "cashflow",
    ],
    "ethics_compliance": [
        "operational_costs",
        "revenue_profit",
    ],
    "data_privacy_cybersecurity": [
        "capex_opex",
        "operational_costs",
    ],
    "tax_transparency": [
        "revenue_profit",
        "cashflow",
    ],

    # ---------------------------------------------------------------
    # UMWELT: ABFALL / WASSER / BIODIVERSITÄT / VERSCHMUTZUNG
    # ---------------------------------------------------------------
    "waste_circularity": [
        "operational_costs",
        "capex_opex",
    ],
    "water": [
        "operational_costs",
        "capex_opex",
    ],
    "biodiversity": [
        "capex_opex",
    ],
    "pollution_air_soil": [
        "operational_costs",
        "capex_opex",
    ],

    # ---------------------------------------------------------------
    # PRODUKTE / KUNDEN / INNOVATION
    # ---------------------------------------------------------------
    "product_responsibility": [
        "operational_costs",
        "revenue_profit",
    ],
    "customer": [
        "revenue_profit",
        "cashflow",
    ],
    "innovation_digitalization": [
        "capex_opex",
        "operational_costs",
    ],

    # ---------------------------------------------------------------
    # COMMUNITY / ENGAGEMENT / SPENDEN
    # ---------------------------------------------------------------
    "community": [
        "operational_costs",
        "cashflow",
    ],
}
