import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(page_title="Campus Energy Agent", layout="wide")
st.title("⚡ Utility-Based Energy Optimization Dashboard")

DATA_PATH = "data/processed/final_ai_actions.csv"
df = pd.read_csv(DATA_PATH)
actions = ["Low", "Medium", "High"]

st.sidebar.header("Adjust Utility Weights")
w1 = st.sidebar.slider("Comfort Weight", 0.0, 2.0, 1.3, 0.1)
w2 = st.sidebar.slider("Cost Weight", 0.0, 2.0, 0.6, 0.1)
w3 = st.sidebar.slider("Emission Weight", 0.0, 2.0, 0.25, 0.1)


def compute_utility(row, action):
    probs = {"low": 0.2, "medium": 0.5, "high": 0.3}
    total = 0.0
    for state, p in probs.items():
        energy = row["pred_energy"]
        if state == "low":
            energy *= 0.85
        elif state == "high":
            energy *= 1.15

        energy *= {"Low": 0.90, "Medium": 1.00, "High": 1.10}[action]

        base = max(0.0, 100 - abs(row["temperature"] - 24) * 4)
        comfort = min(110.0, base + {"Low": 0, "Medium": 5, "High": 10}[action])
        cost = energy * 0.01
        emission = energy * 0.0004

        comfort_norm = comfort / 110
        cost_norm = cost / max(1.0, row["pred_energy"] * 0.02)
        emission_norm = emission / max(1.0, row["pred_energy"] * 0.0008)

        utility = (
            w1 * comfort_norm
            - w2 * cost_norm
            - w3 * emission_norm
        )
        total += p * utility
    return total


# Date/hour filters.
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
df["day"] = df["day"].astype(str)
dates = sorted(df["day"].dropna().unique())
selected_day = st.sidebar.selectbox("Select Day", dates)
day_df = df[df["day"] == selected_day]
hour = st.sidebar.slider(
    "Select Hour",
    min_value=0,
    max_value=23,
    value=12,
    format="%02d:00",
)

df_filtered = day_df[day_df["hour"] == hour].copy()

if df_filtered.empty:
    nearest_index = (day_df["hour"] - hour).abs().idxmin()
    df_filtered = day_df.loc[[nearest_index]].copy()
    observed_hour = int(df_filtered.iloc[0]["hour"])
    st.info(
        f"No measurement is available at {hour:02d}:00. "
        f"Showing the nearest observation from {observed_hour:02d}:00 instead."
    )

# Re-score the displayed observations with the current sidebar weights. The
# stored scores were generated with fixed weights and can be stale in this view.
df_filtered["action_utilities"] = df_filtered.apply(
    lambda observation: {
        action: compute_utility(observation, action) for action in actions
    },
    axis=1,
)
df_filtered["best_action"] = df_filtered["action_utilities"].apply(
    lambda scores: max(scores, key=scores.get)
)
df_filtered["best_utility"] = df_filtered.apply(
    lambda observation: observation["action_utilities"][observation["best_action"]],
    axis=1,
)

st.subheader(f"📊 Data Snapshot - {selected_day}, Hour {hour}")
st.dataframe(
    df_filtered[[
        "timestamp", "temperature", "humidity", "occupancy",
        "room_type", "energy", "pred_energy", "best_action", "best_utility"
    ]],
    use_container_width=True,
)

row = df_filtered.iloc[0]
action_utils = row["action_utilities"]
best_action = row["best_action"]
best_utility = row["best_utility"]

st.success(f"Recommended Action: {best_action}")
st.metric("Best Utility Score", round(best_utility, 3))

st.subheader("Utility per Action")
df_plot = pd.DataFrame({"Action": actions, "Utility": [action_utils[a] for a in actions]}).set_index("Action")
st.bar_chart(df_plot)

st.subheader("📈 Predicted Energy Trend")
day_df = df[df["day"] == selected_day]
energy_trend = day_df.groupby("hour")["pred_energy"].mean()
st.line_chart(energy_trend)

st.subheader("🧠 Bayesian Decision Network (Live)")
pred = float(row["pred_energy"])
scale = max(pred, 1.0)
p_low = max(0.1, 1 - pred / (2 * scale))
p_high = min(0.7, pred / (2 * scale))
p_medium = max(0.0, 1 - (p_low + p_high))
probs = {
    "Low": round(float(p_low), 2),
    "Medium": round(float(p_medium), 2),
    "High": round(float(p_high), 2),
}

G = nx.DiGraph()
temp_node = f"Temperature\n{row['temperature']:.1f}°C"
hum_node = f"Humidity\n{row['humidity']:.1f}%"
occ_node = f"Occupancy proxy\n{int(row['occupancy'])} active"
energy_node = f"Energy State\n{probs}"
decision_node = f"Decision\n{best_action}"
G.add_nodes_from([temp_node, hum_node, occ_node, energy_node, decision_node])
G.add_edges_from([
    (temp_node, energy_node),
    (hum_node, energy_node),
    (occ_node, energy_node),
    (energy_node, decision_node),
])

fig, ax = plt.subplots(figsize=(9, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=3500, font_size=9, ax=ax)
st.pyplot(fig)

st.subheader("📌 Insight")
st.write(
    "The model predicts real building power demand from sensor conditions. "
    "The utility agent then evaluates Low/Medium/High HVAC-control scenarios. "
    "Sidebar weights change the decision without changing the trained model."
)

st.caption(
    "Source: University of Sharjah M5 smart-building IoT dataset. "
    "Occupancy shown here is a motion-sensor activity proxy, not a measured headcount."
)
