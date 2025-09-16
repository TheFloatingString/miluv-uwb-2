# UWB NLOS experiments

The following repository contains the datasets and code used to run the NLOS detection and obstacle identification experiments.

## Quickstart

```bash
pip install uv  # installs uv as a package manager
uv sync         # installs all the dependencies
```

All the dataset splits are accessible under the `/data` folder.

All the scripts are accessible under the `/scripts` folder, and can be run with the following command:

```bash
uv run scripts/<name of script> <followed by additional arguments>
```

The `scripts/sh` subfolder contains examples of valid scripts and the expected arguments schema.
