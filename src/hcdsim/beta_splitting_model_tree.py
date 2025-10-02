#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_tree.py - Generate phylogenetic trees using the Beta Splitting Model

This script implements a modified Beta Splitting Model for generating evolutionary trees
with variable numbers of children per node.
"""

import numpy as np
import random
import argparse
from collections import deque
import matplotlib.pyplot as plt
import networkx as nx
import json
import sys
import os

# Increase recursion limit to handle deeper trees
sys.setrecursionlimit(10000)

class TreeNode:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.maternal_cnvs = []
        self.paternal_cnvs = []
        self.maternal_fasta = None
        self.paternal_fasta = None
        self.fq1 = None
        self.fq2 = None
        self.fasta = None
        self.maternal_fasta_length = 0
        self.paternal_fasta_length = 0
        self.parent = None
        self.ratio = None
        self.cell_no = None
        self.depth = None
        self.changes = []
        self.id = None
        self.edge_length = 0
        self.interval = None  # Store the interval as [start, end]

    def to_dict(self):
        """Convert the TreeNode and its children to a dictionary."""
        return {
            "name": self.name,
            "children": [child.to_dict() for child in self.children],
            "maternal_cnvs": self.maternal_cnvs,
            "paternal_cnvs": self.paternal_cnvs,
            "maternal_fasta": self.maternal_fasta,
            "paternal_fasta": self.paternal_fasta,
            "fq1": self.fq1,
            "fq2": self.fq2,
            "fasta": self.fasta,
            "maternal_fasta_length": self.maternal_fasta_length,
            "paternal_fasta_length": self.paternal_fasta_length,
            "ratio": self.ratio,
            "cell_no": self.cell_no,
            "depth": self.depth,
            "changes": self.changes,
            "id": self.id,
            "edge_length": self.edge_length,
            "interval": self.interval
        }

    @staticmethod
    def from_dict(data, parent=None):
        """Restore a TreeNode from a dictionary, setting parent references."""
        node = TreeNode(data["name"])
        node.maternal_cnvs = data.get("maternal_cnvs", [])
        node.paternal_cnvs = data.get("paternal_cnvs", [])
        node.maternal_fasta = data.get("maternal_fasta", None)
        node.paternal_fasta = data.get("paternal_fasta", None)
        node.fq1 = data.get("fq1", None)
        node.fq2 = data.get("fq2", None)
        node.fasta = data.get("fasta", None)
        node.maternal_fasta_length = data.get("maternal_fasta_length", 0)
        node.paternal_fasta_length = data.get("paternal_fasta_length", 0)
        node.ratio = data.get("ratio", None)
        node.cell_no = data.get("cell_no", None)
        node.depth = data.get("depth", None)
        node.changes = data.get("changes", [])
        node.id = data.get("id", None)
        node.edge_length = data.get("edge_length", 0)
        node.interval = data.get("interval", None)
        node.parent = parent
        node.children = [TreeNode.from_dict(child, node) for child in data.get("children", [])]
        return node

def set_random_seed(seed):
    """Set random seed for reproducibility."""
    if seed is None:
        # If no seed provided, generate a random one and save it
        seed = int.from_bytes(os.urandom(4), byteorder="little")
    
    # Set seeds for both random and numpy
    random.seed(seed)
    np.random.seed(seed)
    
    return seed

def draw_tree(node, pos=None, level=0, width=2., vert_gap=0.2, xcenter=0.5):
    """Calculate positions for drawing a tree with multiple children per node"""
    if pos is None:
        pos = {node.name: (xcenter, 1 - level * vert_gap)}
    else:
        pos[node.name] = (xcenter, 1 - level * vert_gap)
    
    if node.children:
        # Calculate the width for each child
        child_count = len(node.children)
        dx = width / child_count
        
        # Calculate the starting position for the leftmost child
        start_x = xcenter - width/2 + dx/2
        
        # Position each child
        for i, child in enumerate(node.children):
            child_x = start_x + i * dx
            pos = draw_tree(child, pos=pos,
                          level=level + 1,
                          width=dx, xcenter=child_x)
    return pos

def draw_tree_to_pdf(tree, outfile):
    """Draw tree structure to a PDF file"""
    # Creating a graph for drawing
    graph = nx.DiGraph()
    graph.add_node(tree.name)
    
    def add_edges(node):
        for child in node.children:
            graph.add_edge(node.name, child.name)
            add_edges(child)
    
    add_edges(tree)

    # Calculating positions of nodes
    pos = draw_tree(tree)

    # Prepare node labels with cell count information
    node_labels = {}
    for node_name in graph.nodes():
        for node in collect_all_nodes(tree, mode=1):
            if node.name == node_name and node.cell_no is not None:
                node_labels[node_name] = f"{node_name}\n({node.cell_no} cells)"
                break
    
    # Drawing the tree using matplotlib
    plt.figure(figsize=(20, 10))
    
    # Draw edges
    nx.draw_networkx_edges(graph, pos, arrows=False)
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color="skyblue")
    
    # Draw labels (only once, with cell count information)
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=8, font_color="black")
    
    # Save the tree graph to a PDF file
    plt.savefig(outfile, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()

def cal_tree_depth(tree):
    """Calculate the depth of a tree iteratively to avoid recursion depth issues"""
    if not tree:
        return 0
    
    max_depth = 0
    queue = [(tree, 1)]  # (node, depth) pairs
    
    while queue:
        node, depth = queue.pop(0)
        max_depth = max(max_depth, depth)
        
        for child in node.children:
            queue.append((child, depth + 1))
    
    return max_depth

def count_nodes(tree):
    """Count the total number of nodes in the tree iteratively"""
    if not tree:
        return 0
        
    count = 0
    queue = [tree]
    
    while queue:
        node = queue.pop(0)
        count += 1
        queue.extend(node.children)
    
    return count

def update_depths(root):
    """Update depth values for all nodes iteratively"""
    queue = [(root, 0)]  # (node, depth) pairs
    while queue:
        current, depth = queue.pop(0)
        current.depth = depth
        for child in current.children:
            queue.append((child, depth + 1))

def build_balanced_tree(num_clones, alpha, beta, max_children=5, treedepth=None, treedepthsigma=0.5, balance_factor=0.8):
    """
    Build a balanced tree using a modified Beta Splitting Model.
    
    Parameters:
    - num_clones: Target number of clones (excluding normal) to generate
    - alpha: First parameter of the Beta distribution (higher values = more balanced)
    - beta: Second parameter of the Beta distribution (higher values = more balanced)
    - max_children: Maximum number of children a node can have
    - treedepth: Target tree depth
    - treedepthsigma: Standard deviation for tree depth
    - balance_factor: Controls how strongly to enforce balance (0-1)
    
    Returns:
    - Root node of the constructed tree
    """
    # Create the root of the tree
    root = TreeNode("normal")
    root.interval = [0, 1]
    root.id = 0
    root.depth = 0
    
    # Edge lengths
    n = num_clones * 2  # Approximate the number of edges needed
    ti = np.random.exponential(1, n)
    
    # Normalize branch lengths
    total_length = sum(ti)
    ti = [t / total_length for t in ti]
    
    # Parameters for tree generation
    n_splits = num_clones  # Maximum number of splitting events
    Ui = np.random.uniform(0.0, 1.0, n_splits)  # For selecting nodes to split
    
    # Keep track of all nodes and the next ID to assign
    all_nodes = [root]
    leaves = [root]
    node_id = 0
    
    # Calculate target depth for a balanced tree with num_clones clones
    # For a balanced tree with branching factor b and L clones: depth ≈ log_b(L)
    avg_branching = min(max_children, 3)  # Estimate average branching factor
    target_balanced_depth = max(2, int(np.log(num_clones) / np.log(avg_branching)) + 1)
    
    if treedepth is None:
        treedepth = target_balanced_depth
    
    # Generate tree by splitting leaves
    while node_id < num_clones and leaves:
        # Sort leaves by depth (ascending) to prioritize shallower nodes
        # This helps maintain balance by developing all branches at similar rates
        leaves.sort(key=lambda node: node.depth if node.depth is not None else 0)
        
        # Use balance factor to determine whether to pick from shallowest nodes
        # or use the random selection
        if random.random() < balance_factor:
            # Select from the shallowest nodes to promote balance
            min_depth = leaves[0].depth if leaves[0].depth is not None else 0
            shallowest_leaves = [leaf for leaf in leaves if (leaf.depth is None) or (leaf.depth <= min_depth + 1)]
            leaf_to_split = random.choice(shallowest_leaves)
        else:
            # Select a leaf using Beta-splitting model
            leaf_idx = int(Ui[node_id % len(Ui)] * len(leaves))
            leaf_to_split = leaves[leaf_idx]
        
        leaves.remove(leaf_to_split)
        
        # Determine number of children based on current depth
        # Deeper nodes have fewer children to maintain balance
        current_depth = leaf_to_split.depth if leaf_to_split.depth is not None else 0
        depth_factor = max(0.1, 1.0 - (current_depth / treedepth))
        
        # Calculate number of children
        if max_children > 2:
            # Use depth factor to adjust number of children
            # Deeper nodes get fewer children
            max_allowed = max(2, int(max_children * depth_factor))
            num_children = min(2 + np.random.geometric(0.5), max_allowed)
            
            # Make sure we don't exceed target num_clones
            num_children = min(num_children, num_clones - node_id + 1)
        else:
            num_children = 2
            
        # If we're near the target, adjust children count to hit exactly num_clones
        remaining = num_clones - node_id
        if remaining < num_children:
            num_children = remaining + 1  # +1 because we'll create exactly remaining new nodes
        
        # Generate evenly spaced split points for more balance
        if num_children > 2:
            if random.random() < balance_factor:
                # Evenly spaced splits for balance
                split_points = [i/(num_children) for i in range(1, num_children)]
            else:
                # Beta distribution for some variability
                split_points = []
                for _ in range(num_children - 1):
                    beta_sample = np.random.beta(alpha, beta)
                    split_points.append(beta_sample)
                split_points.sort()
        else:
            # For binary split, use Beta distribution with high alpha/beta for balance
            split_points = [np.random.beta(alpha, beta)]
        
        # Create intervals for children
        start, end = leaf_to_split.interval
        interval_size = end - start
        
        # Convert relative split points to absolute positions
        abs_split_points = [start + point * interval_size for point in split_points]
        
        # Create array with all split boundaries
        boundaries = [start] + abs_split_points + [end]
        
        # Create children
        for j in range(len(boundaries) - 1):
            node_id += 1
            child = TreeNode(f"clone{node_id}")
            child.interval = [boundaries[j], boundaries[j+1]]
            child.parent = leaf_to_split
            child.id = node_id
            child.edge_length = ti[node_id % len(ti)]
            child.depth = leaf_to_split.depth + 1
            
            leaf_to_split.children.append(child)
            all_nodes.append(child)
            leaves.append(child)
            
            # Stop if we've reached the target number of clones
            if node_id >= num_clones:
                break
    
    # Update depths for all nodes
    update_depths(root)
    
    # Balance the tree by pruning deep branches if needed
    if balance_factor > 0.5:
        balance_tree_depths(root)
    
    # Check if tree depth meets target
    if treedepth is not None:
        actual_depth = cal_tree_depth(root)
        if abs(actual_depth - treedepth) > treedepthsigma:
            # Try again with different parameters if depth is not within acceptable range
            return build_balanced_tree(num_clones, alpha, beta, max_children, treedepth, treedepthsigma, balance_factor)
    
    return root

def balance_tree_depths(root):
    """
    Balance a tree by ensuring all leaves are at similar depths.
    This is done by pruning some deep branches and expanding shallow ones.
    """
    # Get all leaves
    leaves = []
    queue = [root]
    while queue:
        node = queue.pop(0)
        if not node.children:
            leaves.append(node)
        else:
            queue.extend(node.children)
    
    # Calculate average leaf depth
    depths = [leaf.depth for leaf in leaves if leaf.depth is not None]
    avg_depth = sum(depths) / len(depths) if depths else 0
    
    # Identify leaves that are much deeper than average
    deep_leaves = [leaf for leaf in leaves if leaf.depth is not None and leaf.depth > avg_depth + 1]
    
    # Prune deep branches by removing some children from their parents
    for leaf in deep_leaves:
        # Find a parent that's close to the average depth
        current = leaf
        target_depth = int(avg_depth)
        
        # Go up the tree until we reach the target depth
        while current.parent and current.depth > target_depth + 1:
            current = current.parent
        
        # If this node has multiple children, we can remove some
        if len(current.children) > 1:
            # Keep only some children to reduce depth
            current.children = current.children[:1]
    
    # Update depths after pruning
    update_depths(root)

def build_multifurcating_tree(num_clones, alpha, beta, max_children=5, treedepth=None, treedepthsigma=0.5):
    """
    Build a tree using a modified Beta Splitting Model allowing multiple children per node.
    
    Parameters:
    - num_clones: Target number of clones (excluding normal) to generate
    - alpha: First parameter of the Beta distribution
    - beta: Second parameter of the Beta distribution
    - max_children: Maximum number of children a node can have (default: 5)
    - treedepth: Target tree depth (if specified)
    - treedepthsigma: Standard deviation for tree depth
    
    Returns:
    - Root node of the constructed tree
    """
    # Create the root of the tree
    root = TreeNode("normal")
    root.interval = [0, 1]
    root.id = 0
    
    # Edge lengths
    n = num_clones * 2  # Approximate the number of edges needed
    ti = np.random.exponential(1, n)
    
    # Normalize branch lengths
    total_length = sum(ti)
    ti = [t / total_length for t in ti]
    
    # Parameters for tree generation
    n_splits = num_clones  # Maximum number of splitting events
    Ui = np.random.uniform(0.0, 1.0, n_splits)  # For selecting nodes to split
    
    # Keep track of all nodes and the next ID to assign
    all_nodes = [root]
    leaves = [root]
    node_id = 0
    
    # Generate tree by splitting leaves
    while node_id < num_clones and leaves:
        # Select a leaf to split
        leaf_idx = int(Ui[node_id % len(Ui)] * len(leaves))
        leaf_to_split = leaves[leaf_idx]
        leaves.remove(leaf_to_split)
        
        # Determine number of children (2 to max_children)
        # Use a distribution that favors fewer children (more realistic)
        if max_children > 2:
            # Geometric distribution favors smaller values
            num_children = min(2 + np.random.geometric(0.5), max_children)
            
            # Make sure we don't exceed target num_clones
            num_children = min(num_children, num_clones - node_id + 1)
        else:
            num_children = 2
            
        # If we're near the target, adjust children count to hit exactly num_clones
        remaining = num_clones - node_id
        if remaining < num_children:
            num_children = remaining + 1  # +1 because we'll create exactly remaining new nodes
        
        # Generate split points using Beta distribution
        if num_children > 2:
            # For multiple splits, we need to generate sorted split points
            split_points = []
            for _ in range(num_children - 1):
                beta_sample = np.random.beta(alpha+1, beta+1)
                split_points.append(beta_sample)
            split_points.sort()
        else:
            # For binary split, just use one Beta sample
            split_points = [np.random.beta(alpha+1, beta+1)]
        
        # Create intervals for children
        start, end = leaf_to_split.interval
        interval_size = end - start
        
        # Convert relative split points to absolute positions
        abs_split_points = [start + point * interval_size for point in split_points]
        
        # Create array with all split boundaries
        boundaries = [start] + abs_split_points + [end]
        
        # Create children
        for j in range(len(boundaries) - 1):
            node_id += 1
            child = TreeNode(f"clone{node_id}")
            child.interval = [boundaries[j], boundaries[j+1]]
            child.parent = leaf_to_split
            child.id = node_id
            child.edge_length = ti[node_id % len(ti)]
            
            leaf_to_split.children.append(child)
            all_nodes.append(child)
            leaves.append(child)
            
            # Stop if we've reached the target number of clones
            if node_id >= num_clones:
                break
    
    # Update depths for all nodes
    update_depths(root)
    
    # Check if tree depth meets target
    if treedepth is not None:
        actual_depth = cal_tree_depth(root)
        if abs(actual_depth - treedepth) > treedepthsigma:
            # Try again with different parameters if depth is not within acceptable range
            return build_multifurcating_tree(num_clones, alpha, beta, max_children, treedepth, treedepthsigma)
    
    return root

def assign_cells_to_all_nodes(root, cell_num):
    """
    Assign cells to all nodes of the tree, not just leaves.
    
    Parameters:
    - root: Root node of the tree
    - cell_num: Total number of cells to distribute
    
    Returns:
    - Updated tree with cell_no assigned to all nodes
    """
    # Collect all nodes in the tree iteratively
    all_nodes = []
    queue = [root]
    
    while queue:
        node = queue.pop(0)
        all_nodes.append(node)
        queue.extend(node.children)
    
    # Assign cells proportionally to all nodes, with more cells to deeper nodes
    # Calculate weighted distribution based on depth
    total_weight = 0
    weights = []
    
    for node in all_nodes:
        # Give higher weight to deeper nodes (more evolved clones)
        weight = 1.0 + node.depth * 0.5
        weights.append(weight)
        total_weight += weight
    
    # Distribute cells based on weights
    remaining_cells = cell_num
    for i, node in enumerate(all_nodes):
        # Calculate cell allocation proportionally
        node_cells = int((weights[i] / total_weight) * cell_num)
        node.cell_no = node_cells
        remaining_cells -= node_cells
    
    # Distribute any remaining cells due to rounding
    for i in range(remaining_cells):
        all_nodes[i % len(all_nodes)].cell_no += 1
    
    return root

def generate_tree_beta(cell_num=10000, num_clones=10, alpha=10.0, beta=10.0, treedepth=4, 
                      treedepthsigma=0.5, max_children=4, balance_factor=0.8, seed=None):
    """
    Generate a random tree using the modified Beta Splitting Model with multifurcation.
    
    Parameters:
    - cell_num: Number of cells to distribute (default: 10000)
    - num_clones: Target number of clones to generate, excluding normal (default: 10)
    - alpha: Alpha parameter for the Beta distribution (default: 10.0)
    - beta: Beta parameter for the Beta distribution (default: 10.0)
    - treedepth: Mean of the tree depth distribution (default: 4)
    - treedepthsigma: Standard deviation of the tree depth distribution (default: 0.5)
    - max_children: Maximum number of children per node (default: 3)
    - balance_factor: Controls tree balance (0-1, higher = more balanced) (default: 0.8)
    - seed: Random seed for reproducibility (default: None)
    
    Returns:
    - Root node of the generated tree
    """
    # Set random seed if provided
    if seed is not None:
        set_random_seed(seed)
    
    # Maximum attempts to generate a tree with the right parameters
    max_attempts = 10
    
    for attempt in range(max_attempts):
        try:
            # Choose between balanced or regular tree based on balance factor
            if balance_factor > 0:
                # Build a balanced tree
                root = build_balanced_tree(
                    num_clones, 
                    alpha, 
                    beta, 
                    max_children,
                    treedepth, 
                    treedepthsigma,
                    balance_factor
                )
            else:
                # Build a regular multifurcating tree
                root = build_multifurcating_tree(
                    num_clones, 
                    alpha, 
                    beta, 
                    max_children,
                    treedepth, 
                    treedepthsigma
                )
            
            # Assign cells to all nodes in the tree
            root = assign_cells_to_all_nodes(root, cell_num)
            
            return root
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {str(e)}")
    
    # If all attempts fail, try with simpler parameters
    print(f"Warning: Could not generate tree after {max_attempts} attempts. Trying with simplified parameters.")
    if balance_factor > 0:
        root = build_balanced_tree(min(5, num_clones), alpha, beta, 3, 3, 0.5, balance_factor)
    else:
        root = build_multifurcating_tree(min(5, num_clones), alpha, beta, 3)
    root = assign_cells_to_all_nodes(root, cell_num)
    return root

def save_tree_to_file(root, filename):
    """Save the tree to a file in JSON format."""
    with open(filename, "w") as file:
        json.dump(root.to_dict(), file, indent=4)

def load_tree_from_file(filename):
    """Load the tree from a JSON file."""
    with open(filename, "r") as file:
        data = json.load(file)
        return TreeNode.from_dict(data)

def tree_to_newick(node):
    """Convert tree to Newick format"""
    if not node.children:
        # If the node has no children, return its name
        result = node.name
    else:
        # If the node has children, recursively process each child
        child_strings = [tree_to_newick(child) for child in node.children]

        # Join child strings with commas and enclose in parentheses
        children_str = ",".join(child_strings)
        result = f"({children_str}){node.name}"

    # Add additional information if available
    if node.ratio:
        result += f"[ratio={node.ratio}]"
    if node.cell_no is not None:
        result += f"[cell_no={node.cell_no}]"

    return result

def main():
    """Main function to parse arguments and generate the tree"""
    parser = argparse.ArgumentParser(description='Generate phylogenetic trees using the Beta Splitting Model')
    
    parser.add_argument('-c', '--cell-num', type=int, default=10000,
                        help='Total number of cells to distribute across the tree (default: 10000)')
    parser.add_argument('-n', '--num-clones', type=int, default=10,
                        help='Number of clones to generate, excluding normal (default: 10)')
    parser.add_argument('-B', '--Beta', type=float, default=10.0,
                        help='The Beta in Beta-splitting model. Higher values = more balanced. (default: 10.0)')
    parser.add_argument('-A', '--Alpha', type=float, default=10.0,
                        help='The Alpha in Beta-splitting model. Higher values = more balanced. (default: 10.0)')
    parser.add_argument('-G', '--treedepth', type=float, default=4,
                        help='The mean of the tree depth distribution. (default: 4)')
    parser.add_argument('-K', '--treedepthsigma', type=float, default=0.5,
                        help='The standard deviation of the tree depth distribution. (default: 0.5)')
    parser.add_argument('-M', '--max-children', type=int, default=3,
                        help='Maximum number of children a node can have. (default: 3)')
    parser.add_argument('-b', '--balance', type=float, default=0.8,
                        help='Balance factor (0-1). Higher values create more balanced trees. (default: 0.8)')
    parser.add_argument('-o', '--output', type=str, default='tree',
                        help='Output file prefix for the tree. (default: "tree")')
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Random seed for reproducible tree generation. If not provided, a random seed will be generated.')
    
    args = parser.parse_args()
    
    # Set the random seed and get the value used (in case it was auto-generated)
    actual_seed = set_random_seed(args.seed)
    
    # Generate the tree
    root = generate_tree_beta(
        args.cell_num,
        args.num_clones,
        args.Alpha,
        args.Beta,
        args.treedepth,
        args.treedepthsigma,
        args.max_children,
        args.balance,
        seed=actual_seed
    )
    
    # Save the tree
    save_tree_to_file(root, f"{args.output}.json")
    
    # Draw the tree
    draw_tree_to_pdf(root, f"{args.output}.pdf")
    
    # Count clones (excluding normal)
    all_nodes = collect_all_nodes(root, 1)
    clone_count = len([node for node in all_nodes if node.name != "normal"])
    leaf_count = len([node for node in all_nodes if not node.children])
    
    # Print tree information
    print(f"Generated tree with seed: {actual_seed}")
    print(f"To reproduce this exact tree, use: -s {actual_seed}")
    print(f"Generated tree with {clone_count} clones (excluding normal).")
    print(f"Tree has {leaf_count} leaf nodes.")
    print(f"Tree depth: {cal_tree_depth(root)}")
    print(f"Total cells: {args.cell_num}")
    
    # Print depth distribution to check balance
    depths = {}
    for node in all_nodes:
        if node.depth not in depths:
            depths[node.depth] = 0
        depths[node.depth] += 1
    
    print("\nDepth distribution:")
    for depth in sorted(depths.keys()):
        print(f"Depth {depth}: {depths[depth]} nodes")
    
    # Print cell distribution
    print("\nCell distribution:")
    for node in all_nodes:
        print(f"{node.name}: {node.cell_no} cells")
    
    # Calculate total cells to verify
    total_cells = sum(node.cell_no for node in all_nodes if node.cell_no is not None)
    print(f"\nTotal cells in tree: {total_cells}")
    
    # Save Newick format
    with open(f"{args.output}.newick", "w") as f:
        f.write(tree_to_newick(root) + ";")
    
    # Save the seed to a metadata file for future reference
    with open(f"{args.output}_metadata.txt", "w") as f:
        f.write(f"Random seed: {actual_seed}\n")
        f.write(f"Alpha: {args.Alpha}\n")
        f.write(f"Beta: {args.Beta}\n")
        f.write(f"Number of clones: {clone_count}\n")
        f.write(f"Balance factor: {args.balance}\n")
        f.write(f"Tree depth: {cal_tree_depth(root)}\n")
        f.write(f"Leaf nodes: {leaf_count}\n")
        f.write(f"Total nodes: {len(all_nodes)}\n")
    
    return root

def collect_all_nodes(root, mode=0):
    """
    Collect all nodes of a tree into an array iteratively.
    :param root: The root node of the tree.
    :param mode: 0 to exclude normal node, 1 to include all nodes
    :return: A list containing nodes in the tree.
    """
    if not root:
        return []
        
    all_nodes = []
    queue = [root]
    
    while queue:
        node = queue.pop(0)
        
        if mode == 0:  # remove normal node
            if node.name != 'normal':
                all_nodes.append(node)
        else:
            all_nodes.append(node)
            
        queue.extend(node.children)
    
    return all_nodes

def update_node_in_tree(root, new_node):
    """
    Update a node in the tree with the same name as the new_node.
    If a match is found, the node in the tree is updated with new_node's attributes.

    :param root: The root of the tree.
    :param new_node: The new node to update.
    :return: The updated root of the tree.
    """
    def dfs(node):
        if node.name == new_node.name:
            # 更新现有节点的所有属性
            node.children = new_node.children
            node.maternal_cnvs = new_node.maternal_cnvs
            node.paternal_cnvs = new_node.paternal_cnvs
            node.maternal_fasta = new_node.maternal_fasta
            node.paternal_fasta = new_node.paternal_fasta
            node.fq1 = new_node.fq1
            node.fq2 = new_node.fq2
            node.fasta = new_node.fasta
            node.maternal_fasta_length = new_node.maternal_fasta_length
            node.paternal_fasta_length = new_node.paternal_fasta_length
            node.parent = new_node.parent
            node.ratio = new_node.ratio
            node.cell_no = new_node.cell_no
            node.depth = new_node.depth
            node.changes = new_node.changes
            return True
        for child in node.children:
            if dfs(child):
                return True
        return False

    dfs(root)
    return root

if __name__ == "__main__":
    main()