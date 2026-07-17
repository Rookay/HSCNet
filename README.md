# Anonymous Core Code

This folder contains a compact, anonymized implementation of the core method
used in the submission. It is intended for method inspection only.

Included:
- `hscnet.py`: the main hierarchical semantic constraint network.
- `losses.py`: semantic alignment, prototype consistency, contrastive, and hierarchy losses.

Omitted intentionally:
- dataset paths and preprocessing scripts;
- subject identifiers and split logic;
- experiment logs, checkpoints, and result aggregation;
- comparison/baseline methods;
- command-line training or evaluation entry points.

The code therefore illustrates the model logic, but is not a directly
reproducible experiment package.
