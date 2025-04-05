# commands/subdetect.py

import pandas as pd
import os
from sklearn.metrics import adjusted_rand_score as adjustedRandIndex, mean_squared_error
from sklearn.metrics import adjusted_mutual_info_score as AMI


def add_subdetect_subparser(subparsers):
    """
    Add the subdetect subcommand to the CLI tool.

    This subcommand is used to calculate the Adjusted Mutual Information (AMI)
    and Adjusted Rand Index (ARI) based on a classification file.
    """
    parser = subparsers.add_parser(
        "subdetect",
        help="Calculate AMI (Adjusted Mutual Information) and ARI (Adjusted Rand Index)."
    )
    parser.add_argument(
        "--classification-files", 
        type=str, 
        nargs = "+",
        required=True, 
        metavar="",
        help="Paths to the classification files, separated by spaces. Each file must "
            "contain 'cell_id' and 'clone_id' columns. The 'clone_id' column represents "
            "classification information, while 'cell_id' combines 'ground_truth_clone_id' "
            "and cell details using an underscore."
    )
    parser.add_argument(
        "--tool-names", 
        type=str, 
        nargs = "+",
        required=True, 
        metavar="",
        help="List of tool names corresponding to the classification files"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./output", 
        metavar="",
        help="Path to the file where the results will be saved, default: './output' "
    )
    return parser

def evaluate_clustering_results(file_path, tool_name="tool_name"):
    """
    Process a CSV file containing clustering data and evaluate clustering performance using ARI and AMI metrics.

    Parameters:
    file_path (str): Path to the input CSV file containing clustering data.
    tool_name (str, optional): Name of the tool used for generating the data. Defaults to "tool_name".

    Returns:
    dict: A dictionary containing the tool name, Adjusted Rand Index (ARI), and Adjusted Mutual Information (AMI).
    """
    data = pd.read_csv(file_path)
    
    # Extract the 'clone_id' column as clusters
    clusters = data['clone_id']
    
    # Extract 'cell_id', split by "_" to get cell clones, and to generate numeric labels
    cell_clones = pd.factorize(data['cell_id'].str.split("_").str[0])[0]
    
    # Compute the Adjusted Rand Index (ARI) between clusters and cell clones
    ari = adjustedRandIndex(clusters, cell_clones)
    
    # Compute the Adjusted Mutual Information (AMI) between clusters and cell clones
    ami = AMI(clusters, cell_clones)
    
    # Return the results as a dictionary
    return {'Tool': tool_name, 'ARI': ari, 'AMI': ami}

# python src/main.py subdetect --classification-files  data\clustering_result.csv  --tool-names chisel

def run(args):
    # Validate input
    if len(args.classification_files) != len(args.tool_names):
        raise ValueError("The number of classification files must match the number of tool names.")
    
    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for path, tool in zip(args.classification_files, args.tool_names):
        result = evaluate_clustering_results(path, tool)
        results.append(result)
    
    result_df = pd.DataFrame(results)
    output_file = os.path.join(args.output_dir, "AMI_ARI_results.csv")
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
