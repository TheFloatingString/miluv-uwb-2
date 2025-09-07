# Experiments

## Overview

1. EWINE vs MILUV using standard preprocessing
2. EWINE vs MILUV using CIR distance scaling
3. MILUV using clock skew measurements (4-vs-2 and 80-20 shuffle)
4. MILUV for obstacle type classification

## Additonal Experiments

1. NLOS on passive UWB
2. Obstacle identification using passive UWB

## Refactoring

- [x] Add f1-score in addition to accuracy
- [ ] Output in tabular format (i.e. csv) that can easily be converted to LaTeX tables

## Tasks

- [ ] full CIR ablations on EWINE (ref. prev. tables) for NLOS
- [ ] full CIR ablations on MILUV (ref. prev. tables) for NLOS
- [ ] full CIR ablations on MILUV (ref. prev. tables) for obstacle type

## 1. EWINE vs MILUV using standard preprocessing

```bash
uv run scripts/run_ewine_vs_miluv.py --model random_forest --dataset "<name of dataset>"
```

Scripts:

```bash
source scripts/sh/ewine_vs_miluv_baseline.sh
source scripts/sh/ewine_vs_miluv_baseline-tabpfn.sh
```

## 2. EWINE vs MILUV using CIR distance scaling

```bash
uv run scripts/run_ewine_vs_miluv.py --model random_forest --scale_cir
```

## 3. MILUV using clock skew measurements (4-vs-2 and 80-20 shuffle)

```bash
uv run scripts/run_skew_classification.py --model random_forest
```

## 4. MILUV for obstacle type classification

```bash
uv run scripts/run_obstacle_type_classification.py --model random_forest
```
