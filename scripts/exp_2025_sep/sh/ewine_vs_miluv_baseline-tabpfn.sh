# EWINE
uv run scripts/run_ewine_vs_miluv.py --model tabpfn --max_10000_rows --source_data EWINE_LOCALIZATION_SET --last_500_cir_cols_only
uv run scripts/run_ewine_vs_miluv.py --model tabpfn --max_10000_rows --source_data EWINE_NLOS_SET --last_500_cir_cols_only

# MILUV
uv run scripts/run_ewine_vs_miluv.py --model tabpfn --source_data MILUV_RANDOM_1_UAV --last_500_cir_cols_only
uv run scripts/run_ewine_vs_miluv.py --model tabpfn --source_data MILUV_RANDOM_3_UAV --last_500_cir_cols_only
uv run scripts/run_ewine_vs_miluv.py --model tabpfn --source_data MILUV_STATIC_1_UAV --last_500_cir_cols_only

# with ablations
uv run scripts/run_ewine_vs_miluv.py --model tabpfn --source_data MILUV_RANDOM_1_UAV --last_500_cir_cols_only --ablations "[\"fft\", \"distance-scale\", \"min-max-scale\"]"
uv run scripts/run_ewine_vs_miluv.py --model tabpfn --source_data MILUV_RANDOM_1_UAV --last_500_cir_cols_only --ablations "[\"fft\", \"ranging-scale\", \"min-max-scale\"]"
