"""Convert the public UoS smart-building sensor files into the schema used by the app.

The UoS release stores energy/electrical and environmental sensors separately. This
script discovers CSV files recursively, normalizes timestamp/value columns, aggregates
energy demand and environmental readings to a common timestamp, and writes:

    data/processed/campus_energy_real.csv

The output deliberately contains NO synthetic AC setting. `occupancy` is a
motion-sensor activity proxy (count of active motion streams at each timestamp),
not a claimed headcount.
"""
from pathlib import Path
import re
import pandas as pd
import numpy as np

RAW_DIR = Path("data/raw")
OUT = Path("data/processed/campus_energy_real.csv")

CATEGORY_MAP = {
    "kwh": "energy_kwh",
    "watt w": "power_w",
    "watt": "power_w",
    "motion": "motion",
    "temperature": "temperature",
    "humidity": "humidity",
}


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def category_for(path):
    parts = [norm(p) for p in path.parts]
    for p in reversed(parts):
        for key, value in CATEGORY_MAP.items():
            if key in p:
                return value
    return None


def find_col(df, candidates):
    normalized = {norm(c): c for c in df.columns}
    for candidate in candidates:
        c = norm(candidate)
        if c in normalized:
            return normalized[c]
    for actual_norm, actual in normalized.items():
        if any(c in actual_norm for c in candidates):
            return actual
    return None


def read_sensor_csv(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None

    tcol = find_col(df, ["timestamp", "time", "datetime", "date"])
    vcol = find_col(df, ["value", "reading", "measurement", "data", "power", "watt", "kwh", "motion", "temperature", "humidity"])
    if not tcol or not vcol:
        return None

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df[tcol], errors="coerce", utc=True),
        "value": pd.to_numeric(df[vcol], errors="coerce"),
    })
    out = out.dropna(subset=["timestamp", "value"])
    if out.empty:
        return None
    return out


def load_category(category):
    frames = []
    for path in RAW_DIR.rglob("*.csv"):
        if category_for(path) != category:
            continue
        df = read_sensor_csv(path)
        if df is not None:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["timestamp", "value"])
    return pd.concat(frames, ignore_index=True)


def resample(frame, value_name, agg="mean"):
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", value_name])
    x = frame[["timestamp", "value"]].copy()
    x = x.set_index("timestamp")["value"]
    if agg == "sum":
        x = x.resample("15min").sum(min_count=1)
    else:
        x = x.resample("15min").mean()
    return x.rename(value_name).reset_index()


def main():
    if not RAW_DIR.exists() or not list(RAW_DIR.rglob("*.csv")):
        raise FileNotFoundError(
            "No raw CSV files found. Run `python download_dataset.py` first."
        )

    power = load_category("power_w")
    if power.empty:
        # Fall back to cumulative kWh if the Watt folder is unavailable.
        power = load_category("energy_kwh")
        if power.empty:
            raise RuntimeError("Could not find Watt W or KWh CSV files in data/raw/")
        power_ts = resample(power, "energy", "mean")
    else:
        # Aggregate device power into building demand at each 15-minute interval.
        power_ts = resample(power, "energy", "sum")

    temp_ts = resample(load_category("temperature"), "temperature")
    hum_ts = resample(load_category("humidity"), "humidity")
    motion = load_category("motion")
    if not motion.empty:
        motion["active"] = (motion["value"] > 0).astype(int)
        motion_ts = motion.set_index("timestamp")["active"].resample("15min").sum().rename("occupancy").reset_index()
    else:
        motion_ts = pd.DataFrame(columns=["timestamp", "occupancy"])

    frames = [power_ts, temp_ts, hum_ts, motion_ts]
    result = frames[0]
    for frame in frames[1:]:
        if not frame.empty:
            result = pd.merge_asof(
                result.sort_values("timestamp"),
                frame.sort_values("timestamp"),
                on="timestamp",
                direction="nearest",
                tolerance=pd.Timedelta("30min"),
            )

    result = result.dropna(subset=["energy", "temperature", "humidity"])
    result["occupancy"] = result.get("occupancy", 0).fillna(0)
    result["occupancy"] = result["occupancy"].clip(lower=0)
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result["hour"] = result["timestamp"].dt.hour
    result["day"] = result["timestamp"].dt.date.astype(str)
    result["day_of_week"] = result["timestamp"].dt.dayofweek
    result["month"] = result["timestamp"].dt.month
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)
    result["room_type"] = "Mixed-use building"

    # Keep the schema intentionally close to the original app.
    result["cost"] = result["energy"] * 0.01
    result["emissions"] = result["energy"] * 0.0004

    cols = [
        "timestamp", "day", "hour", "day_of_week", "month", "is_weekend",
        "temperature", "humidity", "occupancy", "room_type", "energy",
        "cost", "emissions",
    ]
    result = result[cols].sort_values("timestamp").reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)
    print(f"Saved {len(result):,} rows to {OUT}")
    print(result.head())


if __name__ == "__main__":
    main()
