from miluv_uwb_2.dataloader import UwbSingleDatasetLoader, UwbDataset, METADATA_DICT
import numpy as np
import matplotlib.pyplot as plt

labels = ["tx1-tx2", "tx1-tx3", "tx2-tx3", "rx1-rx2", "rx1-rx3", "rx2-rx3", "range", "range_raw", "fpp1", "fpp2", "skew1", "skew2", "tx1", "tx2", "tx3", "rx1", "rx2", "rx3", "tx1-rx1", "tx2-rx2", "tx3-rx3", "is_nlos"]

for dataset_name in METADATA_DICT:
    dl = UwbSingleDatasetLoader(UwbDataset(dataset_name))
    df = dl.dataset.df_single_dataset
    extra_cols = df[["range", "range_raw", "fpp1", "fpp2", "skew1", "skew2", "tx1", "tx2", "tx3", "rx1", "rx2", "rx3"]].values
    tx_rx_diff = np.column_stack([
        df["tx1_raw"].values - df["rx1_raw"].values,
        df["tx2_raw"].values - df["rx2_raw"].values,
        df["tx3_raw"].values - df["rx3_raw"].values,
    ])
    data = np.column_stack([dl.X_data, extra_cols, tx_rx_diff, dl.y_data])
    correlation_matrix = np.corrcoef(data.T)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_title(dataset_name)
    plt.tight_layout()
    plt.savefig(f"heatmap_{dataset_name}.png")
    plt.close(fig)
