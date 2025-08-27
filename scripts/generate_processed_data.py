import pandas as pd 

import io
import pandas as pd 

MILUV_UWB_CIR_FILES = [
    {
        "name": "miluv-random_1-ifo001-uwb_cir",
        "filepath": "data/source_data/miluv/cirObstacles_1_random3_0/ifo001/uwb_cir.csv",
    },
    {
        "name": "miluv-static_1-ifo001-uwb_cir",
        "filepath": "data/source_data/miluv/cirObstaclesOneTag_1_static_0/ifo001/uwb_cir.csv",
    },
    {
        "name": "miluv-random_3-ifo001-uwb_cir",
        "filepath": "data/source_data/miluv/cirObstacles_3_random_0/ifo001/uwb_cir.csv",
    },
    {
        "name": "miluv-random_3-ifo002-uwb_cir",
        "filepath": "data/source_data/miluv/cirObstacles_3_random_0/ifo002/uwb_cir.csv",
    },
    {
        "name": "miluv-random_3-ifo003-uwb_cir",
        "filepath": "data/source_data/miluv/cirObstacles_3_random_0/ifo003/uwb_cir.csv",
    }
]

MILUV_UWB_RANGE_FILES = [
    {
        "name": "miluv-random_1-ifo001-uwb_range",
        "filepath": "data/source_data/miluv/cirObstacles_1_random3_0/ifo001/uwb_range.csv",
    },
    {
        "name": "miluv-static_1-ifo001-uwb_range",
        "filepath": "data/source_data/miluv/cirObstaclesOneTag_1_static_0/ifo001/uwb_range.csv",
    },
    {
        "name": "miluv-random_3-ifo001-uwb_range",
        "filepath": "data/source_data/miluv/cirObstacles_3_random_0/ifo001/uwb_range.csv",
    },
    {
        "name": "miluv-random_3-ifo002-uwb_range",
        "filepath": "data/source_data/miluv/cirObstacles_3_random_0/ifo002/uwb_range.csv",
    },
    {
        "name": "miluv-random_3-ifo003-uwb_range",
        "filepath": "data/source_data/miluv/cirObstacles_3_random_0/ifo003/uwb_range.csv",
    },
]

for file_dict in MILUV_UWB_CIR_FILES:
    df = pd.read_csv(file_dict["filepath"])
    print(df.head())
    for to_id in df.to_id.unique():
        df[df.to_id == to_id].to_csv(f"data/processed_data/{file_dict['name']}_{to_id}.csv", index=False)


for file_dict in MILUV_UWB_RANGE_FILES:
    df = pd.read_csv(file_dict["filepath"])
    print(df.head())
    for to_id in df.to_id.unique():
        df[df.to_id == to_id].to_csv(f"data/processed_data/{file_dict['name']}_{to_id}.csv", index=False)