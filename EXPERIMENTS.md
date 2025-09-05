# Experiments

## Overview

1. EWINE vs MILUV using standard preprocessing
2. EWINE vs MILUV using CIR distance scaling
3. MILUV using clock skew measurements (4-vs-2 and 80-20 shuffle)
4. MILUV for obstacle type classification

## Additonal Experiments

1. NLOS on passive UWB
2. Obstacle identification using passive UWB

## 1. EWINE vs MILUV using standard preprocessing

```bash
uv run scripts/run_ewine_vs_miluv.py --model random_forest
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
