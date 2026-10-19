# Real Dataset Migration

This version uses the **University of Sharjah (UoS) M5 smart-building IoT dataset** instead of the synthetic `campus_energy_big.csv`.

Source: https://zenodo.org/records/12750891
DOI: 10.5281/zenodo.12750891

The source contains energy/electrical measurements plus temperature, humidity and motion/occupancy sensing from a university building. The published description states that data were collected from January 1, 2024 to June 20, 2024 and include areas such as the interdisciplinary lab, kitchen and mailroom.

## Setup

```bash
pip install pandas numpy scikit-learn streamlit networkx matplotlib requests
python download_dataset.py
python preprocess_dataset.py
python ml_model.py
python agent.py
streamlit run app.py
```

## Data pipeline

```text
UoS raw CSVs
    ↓
preprocess_dataset.py
    ↓
data/processed/campus_energy_real.csv
    ↓
ml_model.py (Ridge)
    ↓
campus_energy_pred.csv
    ↓
agent.py (utility/Bayesian-style decision layer)
    ↓
final_ai_actions.csv
    ↓
app.py (Streamlit dashboard)
```

## Important methodological note

The original project generated `ac_level` randomly and then used it to generate synthetic energy. That column is intentionally removed from the real-data ML pipeline. The real dataset provides measured sensor conditions; Low/Medium/High is now a **decision scenario produced by the optimization layer**, not a randomly generated training feature.

`occupancy` is derived from motion-sensor activity and should be described as an **occupancy proxy**, not an exact number of people.

The `cost` and `emissions` columns are modelling estimates based on configurable coefficients; they are not claimed to be measured UoS billing or carbon data.

The source measurements are sparse, so not every date has an observation in every hour. The dashboard's hour slider covers all hours from 00:00 to 23:00; when the selected hour has no measurement, it displays the nearest observation and labels the actual observed hour. The recommended action and utility are recalculated together using the current sidebar weights.
