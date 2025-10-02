# Insider Threat Analysis Demo (Synthetic Data)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

'''
# 1. Generate synthetic log data
'''
np.random.seed(42)

users = [f"user{i}" for i in range(1, 11)]
resources = [f"resource{i}" for i in range(1, 6)]
dates = pd.date_range("2025-01-01", periods=30, freq="D")

data = []
for date in dates:
    for _ in range(np.random.randint(50, 100)):  # events per day
        user = np.random.choice(users)
        resource = np.random.choice(resources)
        event_type = np.random.choice(["login", "access", "download", "failed_login"],
                                      p=[0.4, 0.3, 0.2, 0.1])
        suspicious = 1 if (event_type == "failed_login" or 
                           (event_type == "download" and np.random.rand() < 0.2)) else 0
        data.append([date, user, resource, event_type, suspicious])

df = pd.DataFrame(data, columns=["date", "user", "resource", "event_type", "suspicious"])
print(" Synthetic logs generated")
df.head(10)

'''
# 2. Visualization 1: Suspicious events over time
'''
susp_per_day = df.groupby("date")["suspicious"].sum()

plt.figure(figsize=(10,5))
susp_per_day.plot(marker="o", color="red")
plt.title("Suspicious Events Over Time")
plt.ylabel("Count of Suspicious Events")
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

'''
# 3. Visualization 2: Top users by suspicious events
'''
susp_per_user = df.groupby("user")["suspicious"].sum().sort_values(ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x=susp_per_user.index, y=susp_per_user.values, palette="coolwarm")
plt.title("Top Users by Suspicious Events")
plt.ylabel("Suspicious Event Count")
plt.show()
