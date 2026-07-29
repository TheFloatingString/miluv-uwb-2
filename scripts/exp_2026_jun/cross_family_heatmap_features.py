"""Train on one robot family (miluv/husky), test on the other, using the full
21-feature set from plot_heatmap.py (tx/rx pairwise diffs, range, fpp, skew,
raw tx/rx, tx-rx diffs)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from miluv_uwb_2.dataloader import UwbDataset, METADATA_DICT

FEATURE_LABELS = [
    "tx1-tx2", "tx1-tx3", "tx2-tx3", "rx1-rx2", "rx1-rx3", "rx2-rx3",
    "range", "range_raw", "fpp1", "fpp2", "skew1", "skew2",
    "tx1", "tx2", "tx3", "rx1", "rx2", "rx3",
    "tx1-rx1", "tx2-rx2", "tx3-rx3",
]

HUSKY_DATASETS = ["husky_uwb_los_all", "husky_uwb_nlos_all_nonsevere"]
MILUV_DATASETS = [
    "miluv_cirObstacles_1_random3_0",
    "miluv_cirObstacles_3_random_0",
    "miluv_cirObstaclesOneTag_1_static_0",
]


def build_features(df: pd.DataFrame) -> np.ndarray:
    tx1, tx2, tx3 = df["tx1_raw"].values, df["tx2_raw"].values, df["tx3_raw"].values
    rx1, rx2, rx3 = df["rx1_raw"].values, df["rx2_raw"].values, df["rx3_raw"].values
    diff_feats = np.column_stack([
        tx1 - tx2, tx1 - tx3, tx2 - tx3,
        rx1 - rx2, rx1 - rx3, rx2 - rx3,
    ])
    extra = df[["range", "range_raw", "fpp1", "fpp2", "skew1", "skew2",
                "tx1", "tx2", "tx3", "rx1", "rx2", "rx3"]].values
    tx_rx_diff = np.column_stack([tx1 - rx1, tx2 - rx2, tx3 - rx3])
    return np.column_stack([diff_feats, extra, tx_rx_diff])


def load_family(dataset_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X_parts, y_parts = [], []
    for name in dataset_names:
        assert name in METADATA_DICT, name
        ds = UwbDataset(name)
        df = ds.df_single_dataset
        X_parts.append(build_features(df))
        y_parts.append(df["is_nlos"].values)
    return np.concatenate(X_parts), np.concatenate(y_parts)


def evaluate(train_name, X_train, y_train, test_name, X_test, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== Train: {train_name} ({len(y_train)} rows) -> Test: {test_name} ({len(y_test)} rows) ===")
    print(f"Train class balance (NLOS frac): {y_train.mean():.3f}")
    print(f"Test class balance (NLOS frac):  {y_test.mean():.3f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1:       {f1:.4f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]:\n{cm}")

    top_idx = np.argsort(clf.feature_importances_)[::-1][:5]
    print("Top 5 features:", [(FEATURE_LABELS[i], round(clf.feature_importances_[i], 3)) for i in top_idx])
    return acc, f1, cm


def feature_correlation_matrix(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """21x21 feature (+is_nlos) Pearson correlation matrix, one family, matching plot_heatmap.py style."""
    data = np.column_stack([X, y])
    return np.corrcoef(data.T)


def plot_correlation_heatmaps(corr_miluv: np.ndarray, corr_husky: np.ndarray, out_path: str):
    labels = FEATURE_LABELS + ["is_nlos"]
    corr_diff = corr_husky - corr_miluv

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    panels = [
        ("miluv: feature correlation", corr_miluv, "coolwarm", -1, 1, "miluv feature", "miluv feature"),
        ("husky: feature correlation", corr_husky, "coolwarm", -1, 1, "husky feature", "husky feature"),
        ("husky - miluv (divergence)", corr_diff, "coolwarm", -2, 2, "husky feature", "miluv feature"),
    ]
    for ax, (title, mat, cmap, vmin, vmax, xlabel, ylabel) in zip(axes, panels):
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Cross-family feature correlation: miluv vs husky")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    X_husky, y_husky = load_family(HUSKY_DATASETS)
    X_miluv, y_miluv = load_family(MILUV_DATASETS)

    evaluate("miluv (all)", X_miluv, y_miluv, "husky (all)", X_husky, y_husky)
    evaluate("husky (all)", X_husky, y_husky, "miluv (all)", X_miluv, y_miluv)

    corr_miluv = feature_correlation_matrix(X_miluv, y_miluv)
    corr_husky = feature_correlation_matrix(X_husky, y_husky)

    plot_correlation_heatmaps(corr_miluv, corr_husky, "heatmap_cross_family_confusion_matrix.png")
