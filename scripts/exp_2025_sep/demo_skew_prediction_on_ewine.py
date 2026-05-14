import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

np.random.seed(42)

CHOICE_OF_EWINE = "2018"

if CHOICE_OF_EWINE == "2016":
    df = pd.read_csv("data/source_data/ewine/2016_paper/uwb_dataset_part6.csv")
    print(df.head())
    col_for_range = df["RANGE"].values
    col_for_is_nlos = df["NLOS"].values
    fp_amp_1 = np.sqrt(df["FP_AMP1"].values)
    fp_amp_2 = np.sqrt(df["FP_AMP2"].values)
else:
    df = pd.concat(
        [
            pd.read_csv(
                f"data/source_data/ewine/2018_paper/dataset2_tag_room1_part{i}.csv",
                header=None,
            )
            for i in range(10)
        ]
    )
    print(df.head())

    col_for_range = df.iloc[:, 4]
    col_for_is_nlos = df.iloc[:, 5]
    fp_amp_1 = np.sqrt(df.iloc[:, 11])
    fp_amp_2 = np.sqrt(df.iloc[:, 12])


model = joblib.load("skew_prediction_model.pkl")

# numpy array where the cols are col_for_range, fp_amp_1, fp_amp_2
X_data = np.column_stack((col_for_range, fp_amp_1, fp_amp_2))

y_pred = model.predict(X_data)

X_train, X_test, y_train, y_test = train_test_split(
    y_pred, col_for_is_nlos, test_size=0.2, random_state=42
)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
