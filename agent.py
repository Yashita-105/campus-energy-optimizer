import pandas as pd
import numpy as np

# Load dataset

df = pd.read_csv("campus_energy_big_pred.csv")

actions = ["Low", "Medium", "High"]

# Bayesian Energy States

def energy_probabilities(pred_energy):
    return {
        "low": 0.2,
        "medium": 0.5,
        "high": 0.3
    }

def adjust_energy(pred_energy, state):
    if state == "low":
        return pred_energy * 0.85
    elif state == "high":
        return pred_energy * 1.15
    return pred_energy


# Compute Components

def compute_components(row, energy, action):
    base = 100 - abs(row["temperature"] - 24)

    # Balanced comfort impact
    if action == "High":
        comfort = base + 10
    elif action == "Medium":
        comfort = base + 5
    else:
        comfort = base

    cost = energy * 5
    emission = energy * 0.4

    return comfort, cost, emission


# Expected Utility (FINAL AI)
# -----------------------------
def expected_utility(row, action):
    probs = energy_probabilities(row["pred_energy"])
    total = 0

    for state, p in probs.items():
        energy = adjust_energy(row["pred_energy"], state)

        # Action impact
        if action == "Medium":
            energy += 12
        elif action == "High":
            energy += 25

        comfort, cost, emission = compute_components(row, energy, action)

  
        # NORMALIZATION (BALANCED)
       
        comfort_norm = comfort / 110
        cost_norm = cost / 600
        emission_norm = emission / 250

    
        # Dynamic Weights (FINAL TUNED)
      
        if row["occupancy"] > 40:
            w1, w2, w3 = 1.5, 0.5, 0.2
        elif 18 <= row["hour"] <= 22:
            w1, w2, w3 = 1.2, 0.7, 0.3
        else:
            w1, w2, w3 = 1.3, 0.6, 0.25

        utility = (
            w1 * comfort_norm
            - w2 * cost_norm
            - w3 * emission_norm
        )

        total += p * utility

    return total


# Decision Making

best_actions = []
best_utilities = []

for _, row in df.iterrows():
    action_utils = {a: expected_utility(row, a) for a in actions}
    best_action = max(action_utils, key=action_utils.get)

    best_actions.append(best_action)
    best_utilities.append(action_utils[best_action])

df["best_action"] = best_actions
df["best_utility"] = best_utilities


# Save Output

df.to_csv("final_ai_actions.csv", index=False)

print("FINAL LOCKED AI AGENT RUN SUCCESS")
print("Mixed actions generated (Low/Medium/High)")