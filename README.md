# GreenwashingDetection

Dieses Projekt kombiniert Claim-Erkennung, KPI-Extraktion und LLM-gestützte Bewertung, um mögliche Greenwashing-Tendenzen in Unternehmensberichten aufzuspüren. Die Pipeline liest Nachhaltigkeits- und Geschäftsberichte als PDF, klassifiziert umweltbezogene Aussagen ("Claims"), extrahiert finanzielle Kennzahlen und bildet anschließend Matching-Batches, die von einem LLM beurteilt werden können.

## Hauptfunktionen
- **Claim-Extraktion:** Klassifiziert Sätze aus Nachhaltigkeitsberichten mit einem ClimateBERT-Modell als "yes/no"-Claims.
- **KPI-Extraktion:** Nutzt die OpenRouter-API (OpenAI-kompatibel), um aus Geschäftsberichten finanzielle Kennzahlen zu extrahieren.
- **Clustering & Topics:** Bildet Claim-Cluster mit Sentence-Embeddings und weist ihnen Themen zu.
- **Matching & Evaluation:** Verknüpft Claims mit passenden KPIs, erstellt Batches und übergibt sie optional an ein LLM zur Bewertung.

## Verzeichnisstruktur
```
./data/
  input/
    nachhaltigkeitsberichte/   # Erwartete PDF-Dateien für die Claim-Extraktion
    geschaeftsberichte/        # Erwartete PDF-Dateien für die KPI-Extraktion
  output/
    claims/                    # JSON-Dateien mit extrahierten Claims
    kpis/                      # JSON-Dateien mit extrahierten KPIs
    matching/                  # Matching-Batches (Claims ↔ KPIs)
    results/                   # LLM-basierte Evaluationsausgaben
src/                           # Pipeline-Implementierung (siehe src/main.py)
```


## API- und Umgebungsvariablen
Für die KPI-Extraktion und die LLM-basierte Evaluation werden OpenRouter-Zugangsdaten erwartet (OpenAI-kompatible Schnittstelle). Legen Sie eine `.env` im Projektwurzelverzeichnis an oder setzen Sie die Variablen in Ihrer Shell:
```
OPENROUTER_API_KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1  
OPENROUTER_MODEL=meta-llama/llama-3.1-70b-instruct 
```

## Nutzung über die Kommandozeile
Die Pipeline wird über `src/main.py` gesteuert. Standardmäßig werden alle Schritte ausgeführt; einzelne Schritte lassen sich auswählen.

Beispielaufrufe:
```bash
# Vollständige Pipeline auf dem Standard-Eingabepfad (data/input)
python -m src.main

# Nur Claims und KPIs extrahieren
python -m src.main --steps claims kpis

# Nur Matching und Evaluation auf bereits berechneten JSONs ausführen
python -m src.main --steps matching evaluation

# Anderes Eingabeverzeichnis und Ausgabe in eine Datei schreiben
python -m src.main --input-dir /pfad/zu/daten --steps all --output result.json
```

## Programmatische Schnelltests
Für Notebook- oder Skript-Nutzung stehen Hilfsfunktionen zur Verfügung (siehe `src/main.py`):
- `quick_test_claims(input_dir=None, return_result=False)`
- `quick_test_kpis(input_dir=None, return_result=False)`
- `quick_test_matching(input_dir=None, return_result=False)`
- `quick_test_evaluation(input_dir=None, return_result=False)`
- `quick_test_claims_clustering(claims_output_dir=None, n_clusters=10, return_result=False)`
