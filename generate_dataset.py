import pandas as pd
import numpy as np

np.random.seed(42)

rows = []

ROOM_TYPES = ["Classroom", "Lab", "Office"]
AC_LEVELS = ["Low", "Medium", "High"]

for day in range(1, 366):  # 1 year
    for hour in range(24):
        for room in range(10):
            temperature = np.random.normal(25, 5)
            humidity = np.random.uniform(30, 70)
            occupancy = np.random.randint(0, 60)
            room_type = np.random.choice(ROOM_TYPES)
            ac_level = np.random.choice(AC_LEVELS)

            energy = 20 + 1.8*temperature + 0.5*humidity + 0.8*occupancy
            if ac_level == "Medium":
                energy += 15
            elif ac_level == "High":
                energy += 30
            if room_type == "Lab":
                energy += 20
            elif room_type == "Office":
                energy += 10

            energy += np.random.normal(0, 5)  # noise

            cost = energy * 5
            emissions = energy * 0.4

            rows.append([day, hour, temperature, humidity, occupancy, room_type, ac_level, energy, cost, emissions])

df = pd.DataFrame(rows, columns=[
    "day", "hour", "temperature", "humidity", "occupancy", "room_type",
    "ac_level", "energy", "cost", "emissions"
])

df.to_csv("campus_energy_big.csv", index=False)
print("Big dataset generated:", df.shape)
