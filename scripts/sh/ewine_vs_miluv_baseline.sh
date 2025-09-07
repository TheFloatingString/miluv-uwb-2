# EWINE
uv run scripts\\run_ewine_vs_miluv.py --model svc --subsample 1.0 --source_data EWINE_LOCALIZATION_SET
uv run scripts\\run_ewine_vs_miluv.py --model svc --subsample 1.0 --source_data EWINE_NLOS_SET

uv run scripts\\run_ewine_vs_miluv.py --model random_forest --subsample 1.0 --source_data EWINE_LOCALIZATION_SET
uv run scripts\\run_ewine_vs_miluv.py --model random_forest --subsample 1.0 --source_data EWINE_NLOS_SET

# MILUV
uv run scripts\\run_ewine_vs_miluv.py --model svc --subsample 1.0 --source_data MILUV_RANDOM_1_UAV
uv run scripts\\run_ewine_vs_miluv.py --model svc --subsample 1.0 --source_data MILUV_RANDOM_3_UAV
uv run scripts\\run_ewine_vs_miluv.py --model svc --subsample 1.0 --source_data MILUV_STATIC_1_UAV

uv run scripts\\run_ewine_vs_miluv.py --model random_forest --subsample 1.0 --source_data MILUV_RANDOM_1_UAV
uv run scripts\\run_ewine_vs_miluv.py --model random_forest --subsample 1.0 --source_data MILUV_RANDOM_3_UAV
uv run scripts\\run_ewine_vs_miluv.py --model random_forest --subsample 1.0 --source_data MILUV_STATIC_1_UAV
