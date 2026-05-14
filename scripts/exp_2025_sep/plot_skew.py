import matplotlib.pyplot as plt
import pandas as pd
from miluv_uwb_2.utils import is_nlos_miluv, get_obstacle_type_miluv

df = pd.read_csv("data/source_data/miluv/cirObstacles_1_random3_0/ifo001/uwb_range.csv")
print(df.head())

X_skew_1 = df["skew1"].values
X_skew_2 = df["skew2"].values

color_map = {0: "blue", 1: "orange", 3: "green", 4: "purple"}

plt.scatter(
    X_skew_1,
    X_skew_2,
    c=[color_map[i] for i in df["to_id"].apply(get_obstacle_type_miluv).values],
    s=0.1,
    alpha=1,
)
plt.show()
