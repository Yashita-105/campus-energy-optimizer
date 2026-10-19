import pandas as pd

INPUT_PATH = "data/processed/campus_energy_pred.csv"
OUTPUT_PATH = "data/processed/final_ai_actions.csv"

df = pd.read_csv(INPUT_PATH)
actions = ["Low", "Medium", "High"]


def energy_probabilities():
    # Lightweight belief model retained from the original project.
    return {"low": 0.2, "medium": 0.5, "high": 0.3}


def adjust_energy(pred_energy, state):
    if state == "low":
        return pred_energy * 0.85
    if state == "high":
        return pred_energy * 1.15
    return pred_energy


def compute_components(row, energy, action):
    # Comfort proxy: 24 C is treated as the center of the preferred range.
    base = max(0.0, 100 - abs(row["temperature"] - 24) * 4)

    # More HVAC effort is assumed to improve thermal comfort.
    comfort_bonus = {"Low": 0, "Medium": 5, "High": 10}[action]
    comfort = min(110.0, base + comfort_bonus)

    # These are modelling assumptions, not measurements from the source dataset.
    cost = energy * 0.01
    emission = energy * 0.0004
    return comfort, cost, emission


def expected_utility(row, action, weights):
    probs = energy_probabilities()
    total = 0.0

    for state, p in probs.items():
        energy = adjust_energy(row["pred_energy"], state)

        # Control-action scenario. The real dataset has no artificial AC-level column.
        energy *= {"Low": 0.90, "Medium": 1.00, "High": 1.10}[action]

        comfort, cost, emission = compute_components(row, energy, action)
        comfort_norm = comfort / 110
        cost_norm = cost / max(1.0, row["pred_energy"] * 0.02)
        emission_norm = emission / max(1.0, row["pred_energy"] * 0.0008)

        utility = (
            weights[0] * comfort_norm
            - weights[1] * cost_norm
            - weights[2] * emission_norm
        )
        total += p * utility

    return total


# Default weights match the original dashboard.
weights = (1.3, 0.6, 0.25)

best_actions = []
best_utilities = []

for _, row in df.iterrows():
    action_utils = {
        action: expected_utility(row, action, weights)
        for action in actions
    }
    best_action = max(action_utils, key=action_utils.get)
    best_actions.append(best_action)
    best_utilities.append(action_utils[best_action])

df["best_action"] = best_actions
df["best_utility"] = best_utilities

df.to_csv(OUTPUT_PATH, index=False)
print("AI agent completed")
print(f"Saved to {OUTPUT_PATH}")
print(df["best_action"].value_counts())
