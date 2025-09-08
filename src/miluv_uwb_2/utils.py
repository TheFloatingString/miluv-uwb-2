import pandas as pd
import numpy as np

uwb_constellation_pos = {
    0: {
        0: [3.273827392578125, 3.46404736328125, 1.8093309326171875],
        1: [3.186386962890625, 0.27394485473632812, 1.5884853515625],
        2: [2.850500244140625, -2.923056884765625, 1.89742041015625],
        3: [-2.497634521484375, -3.5018203125, 1.7730911865234375],
        4: [-2.95793310546875, 0.6128419189453125, 1.65714208984375],
        5: [-2.734676513671875, 3.65854248046875, 1.890254638671875],
    }
}


def is_nlos_miluv(tag_id) -> int:
    # NLOS
    if tag_id in [1, 3, 4]:
        return 1
    # LOS
    else:
        return 0


def get_obstacle_type_miluv(tag_id) -> int:
    if tag_id in [1, 3, 4]:
        return tag_id
    else:
        return 0


def get_dist_for_row(drone_x, drone_y, drone_z, anchor_id):
    if anchor_id not in uwb_constellation_pos[0]:
        return np.nan
    anchor_pos = uwb_constellation_pos[0][anchor_id]
    return np.sqrt(
        (anchor_pos[0] - drone_x) ** 2
        + (anchor_pos[1] - drone_y) ** 2
        + (anchor_pos[2] - drone_z) ** 2
    )


def get_gt_dist_drone_to_anchor(cir_df, mocap_df) -> pd.DataFrame:
    miluv_df = pd.merge_asof(
        cir_df,
        mocap_df,
        on="timestamp",
        direction="nearest",
    )

    miluv_df["gt_distance_mocap"] = miluv_df.apply(
        lambda row: get_dist_for_row(
            row["pose.position.x"],
            row["pose.position.y"],
            row["pose.position.z"],
            row["to_id"],
        ),
        axis=1,
    )
    miluv_df.dropna(subset=["gt_distance_mocap"], inplace=True)

    print(miluv_df.head())
    return miluv_df
