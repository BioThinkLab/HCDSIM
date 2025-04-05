# hcdbench subdetect

`hcdbench subdetect` command to calculate AMI (Adjusted Mutual Information) and ARI (Adjusted Rand Index).

## Overview

This script implements a subcommand called `subdetect` that calculates the **Adjusted Mutual Information (AMI)** and **Adjusted Rand Index (ARI)** based on classification files. These metrics are used to evaluate the performance of clustering tools across different tasks.

## Parameters
```shell
usage: hcdbench subdetect [-h] --classification-files  [...] --tool-names  [...] [--output-dir]

options:
  -h, --help            show this help message and exit
  --classification-files  [ ...]
                        Paths to the classification files, separated by spaces. Each file must contain 'cell_id' and 'clone_id' columns. The 'clone_id' column
                        represents classification information, while 'cell_id' combines 'ground_truth_clone_id' and cell details using an underscore.
  --tool-names  [ ...]  List of tool names corresponding to the classification files
  --output-dir          Path to the file where the results will be saved, default: './output'
```

## Input

The script requires the following inputs:

1. **Classification files**:
   - Multiple file paths can be provided, separated by spaces.
   - Each file must contain the following two columns:
     - `cell_id`: In the format "ground_truth_clone_id_cell_name", e.g., "cloneA_cell1".
     - `clone_id`: Represents the classification information.
2. **Tool Names**:
   - A list of names corresponding to the tools used to generate the classification files.
3. **Output Directory**:
   - Optional parameter, default path is `./output`.

## Output

- The results will be saved as a CSV file named `AMI_ARI_results.csv`, which is stored by default in the `./output` directory.
- The output will contain:
  - Tool Name (`Tool`)
  - Adjusted Rand Index (ARI)
  - Adjusted Mutual Information (AMI)

## Example Usage

```shell
hcdbench subdetect \
   --classification-files data/clustering_result.csv \
   --tool-names chisel
```