from miluv_uwb_2.dataloader import UwbDataset
import numpy as np
import matplotlib.pyplot as plt

FEATURE_LABELS = [
    "tx1-tx2", "tx1-tx3", "tx2-tx3",
    "rx1-rx2", "rx1-rx3", "rx2-rx3",
    "range", "range_raw", "fpp1", "fpp2", "skew1", "skew2",
    "tx1", "tx2", "tx3", "rx1", "rx2", "rx3",
    "tx1-rx1", "tx2-rx2", "tx3-rx3",
]
CONFIGS = [
    {"nlos_dataset": "husky_uwb_nlos_2_3_11", "tags": [2, 3, 11]},
    {"nlos_dataset": "husky_uwb_nlos_3_5_11", "tags": [3, 5, 11]},
]


def build_features(df):
    tx1, tx2, tx3 = df["tx1_raw"].values, df["tx2_raw"].values, df["tx3_raw"].values
    rx1, rx2, rx3 = df["rx1_raw"].values, df["rx2_raw"].values, df["rx3_raw"].values
    diff_feats = np.column_stack([
        tx1 - tx2, tx1 - tx3, tx2 - tx3,
        rx1 - rx2, rx1 - rx3, rx2 - rx3,
    ])
    extra = df[["range", "range_raw", "fpp1", "fpp2", "skew1", "skew2", "tx1", "tx2", "tx3", "rx1", "rx2", "rx3"]].values
    tx_rx_diff = np.column_stack([tx1 - rx1, tx2 - rx2, tx3 - rx3])
    return np.column_stack([diff_feats, extra, tx_rx_diff])


ds_los = UwbDataset("husky_uwb_los_all")
df_los = ds_los.df_single_dataset
X_los = build_features(df_los)

n_features = len(FEATURE_LABELS)
n_cols = 4
n_rows = (n_features + n_cols - 1) // n_cols

for cfg in CONFIGS:
    nlos_name = cfg["nlos_dataset"]
    tags = cfg["tags"]

    ds_nlos = UwbDataset(nlos_name)
    df_nlos = ds_nlos.df_single_dataset
    X_nlos = build_features(df_nlos)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3.5))
    axes = axes.flatten()
    x = np.arange(len(tags))
    width = 0.35

    for feat_idx, feat_name in enumerate(FEATURE_LABELS):
        ax = axes[feat_idx]
        means_nlos, stds_nlos = [], []
        means_los, stds_los = [], []

        for tag in tags:
            vals_nlos = X_nlos[df_nlos["to_id"].values == tag, feat_idx]
            vals_los = X_los[df_los["to_id"].values == tag, feat_idx]
            means_nlos.append(np.mean(vals_nlos) if len(vals_nlos) > 0 else np.nan)
            stds_nlos.append(np.std(vals_nlos) if len(vals_nlos) > 0 else np.nan)
            means_los.append(np.mean(vals_los) if len(vals_los) > 0 else np.nan)
            stds_los.append(np.std(vals_los) if len(vals_los) > 0 else np.nan)

        ax.bar(x - width / 2, means_nlos, width, yerr=stds_nlos,
               label="NLOS", color="tomato", capsize=4, alpha=0.85)
        ax.bar(x + width / 2, means_los, width, yerr=stds_los,
               label="LOS", color="steelblue", capsize=4, alpha=0.85)
        ax.set_title(feat_name)
        ax.set_xticks(x)
        ax.set_xticklabels([f"tag {t}" for t in tags])
        if feat_idx == 0:
            ax.legend()

    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    tag_str = "_".join(str(t) for t in tags)
    plt.suptitle(f"Mean ± StDev per tag: NLOS ({nlos_name}) vs LOS (husky_uwb_los_all)", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"tag_comparison_{tag_str}.png", dpi=150)
    plt.close(fig)
