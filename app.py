import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(page_title="Campus Energy Agent", layout="wide")
st.title("⚡ Utility-Based Energy Optimization Dashboard")

#Dataset load kar rahe hai
df = pd.read_csv("final_ai_actions.csv")
actions = ["Low", "Medium", "High"]

#Sidebar Logic hai
st.sidebar.header("Adjust Utility Weights")
w1 = st.sidebar.slider("Comfort Weight", 0.0, 2.0, 1.3, 0.1)
w2 = st.sidebar.slider("Cost Weight", 0.0, 2.0, 0.6, 0.1)
w3 = st.sidebar.slider("Emission Weight", 0.0, 2.0, 0.25, 0.1)

#agent.py ka ai logic (functional sidebar)
def compute_utility(row, action):
    probs = {"low": 0.2, "medium": 0.5, "high": 0.3}
    total = 0
    for state, p in probs.items():
        energy = row["pred_energy"]

        if state == "low":
            energy *= 0.85
        elif state == "high":
            energy *= 1.15

        if action == "Medium":
            energy += 12
        elif action == "High":
            energy += 25

        base = 100 - abs(row["temperature"] - 24)

        if action == "High":
            comfort = base + 10
        elif action == "Medium":
            comfort = base + 5
        else:
            comfort = base

        cost = energy * 5
        emission = energy * 0.4

        # normalization (IMPORTANT)
        comfort_norm = comfort / 110
        cost_norm = cost / 600
        emission_norm = emission / 250
        utility = (
            w1 * comfort_norm
            - w2 * cost_norm
            - w3 * emission_norm
        )

        total += p * utility

    return total

#Filters
day = st.sidebar.selectbox("Select Day", sorted(df["day"].unique()))
hour = st.sidebar.slider("Select Hour", 0, 23, 12)

df_filtered = df[(df["day"] == day) & (df["hour"] == hour)]

st.subheader(f"📊 Data Snapshot - Day {day}, Hour {hour}")
st.dataframe(
    df_filtered[
        ["temperature","humidity","occupancy","room_type",
         "ac_level","pred_energy"]
    ]
)

#Decision
row = df_filtered.iloc[0]

action_utils = {a: compute_utility(row, a) for a in actions}
best_action = max(action_utils, key=action_utils.get)
best_utility = action_utils[best_action]

st.success(f" Recommended Action: {best_action}")

st.metric("Best Utility Score", round(best_utility, 3))

#Utility Breakdown
st.subheader("Utility per Action")

import pandas as pd

df_plot = pd.DataFrame({
    "Action": actions,
    "Utility": [action_utils[a] for a in actions]
}).set_index("Action")

st.bar_chart(df_plot)

# Energy Trend
st.subheader("📈 Predicted Energy Trend (Full Day)")

day_df = df[df["day"] == day]
energy_trend = day_df.groupby("hour")["pred_energy"].mean()

st.line_chart(energy_trend)

#BAYES NETWORK
st.subheader("🧠 Bayesian Decision Network (Live)")
row = df_filtered.iloc[0]
# Dynamic probabilities (based on energy)
pred = row["pred_energy"]
# Simple belief model (dynamic)
p_low = max(0.1, 1 - pred/200)
p_high = min(0.7, pred/200)
p_medium = 1 - (p_low + p_high)
probs = {k: round(float(v), 2) for k, v in {
    "Low": p_low,
    "Medium": p_medium,
    "High": p_high
}.items()}
# Build graph
G = nx.DiGraph()
G.add_node("Temperature\n" + str(round(row["temperature"],1)))
G.add_node("Humidity\n" + str(round(row["humidity"],1)))
G.add_node("Occupancy\n" + str(int(row["occupancy"])))
G.add_node("Energy State\n" + str(probs))
G.add_node("Decision\n" + row["best_action"])
G.add_edges_from([
    ("Temperature\n" + str(round(row["temperature"],1)), "Energy State\n" + str(probs)),
    ("Humidity\n" + str(round(row["humidity"],1)), "Energy State\n" + str(probs)),
    ("Occupancy\n" + str(int(row["occupancy"])), "Energy State\n" + str(probs)),
    ("Energy State\n" + str(probs), "Decision\n" + row["best_action"])
])
# Draw
fig, ax = plt.subplots(figsize=(8,6))
pos = nx.spring_layout(G, seed=42)
nx.draw(
    G, pos,
    with_labels=True,
    node_size=3500,
    font_size=9,
    node_color="lightblue",
    ax=ax
)
st.pyplot(fig)

st.subheader("📌 Insight")

st.write("""
- Slider dynamically changes decision priorities
- Agent recomputes expected utility in real-time
- Shows how AI decisions change based on preferences
""")