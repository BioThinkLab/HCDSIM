#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_tree.py - Generate phylogenetic trees using the Beta Splitting Model

This script implements a Beta Splitting Model for generating evolutionary trees
with variable numbers of children per node, using purely iterative approaches.
"""

import numpy as np
import random
import argparse
from collections import deque
import matplotlib.pyplot as plt
import networkx as nx
import json
import os

class TreeNode:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.maternal_cnas = []
        self.paternal_cnas = []
        self.maternal_fasta = None
        self.paternal_fasta = None
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
        self.interval = None
        self.cna_status = None

    def to_dict(self):
        """Convert the TreeNode and its children to a dictionary iteratively."""
        result = {
            "name": self.name,
            "maternal_cnas": self.maternal_cnas,
            "paternal_cnas": self.paternal_cnas,
            "maternal_fasta": self.maternal_fasta,
            "paternal_fasta": self.paternal_fasta,
            "fasta": self.fasta,
            "maternal_fasta_length": self.maternal_fasta_length,
            "paternal_fasta_length": self.paternal_fasta_length,
            "ratio": self.ratio,
            "cell_no": self.cell_no,
            "depth": self.depth,
            "changes": self.changes,
            "id": self.id,
            "edge_length": self.edge_length,
            "interval": self.interval,
            "cna_status": self.cna_status,
            "children": []
        }
        
        # Process children iteratively
        for child in self.children:
            result["children"].append(child.to_dict())
        
        return result

    @staticmethod
    def from_dict(data, parent=None):
        """Restore a TreeNode from a dictionary iteratively."""
        node = TreeNode(data["name"])
        node.maternal_cnas = data.get("maternal_cnas", [])
        node.paternal_cnas = data.get("paternal_cnas", [])
        node.maternal_fasta = data.get("maternal_fasta", None)
        node.paternal_fasta = data.get("paternal_fasta", None)
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
        node.cna_status = data.get("cna_status", None)
        node.parent = parent
        
        # Process children iteratively
        for child_data in data.get("children", []):
            child = TreeNode.from_dict(child_data, node)
            node.children.append(child)
        
        return node

def set_random_seed(seed):
    """Set random seed for reproducibility."""
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="little")
    
    random.seed(seed)
    np.random.seed(seed)
    
    return seed

def draw_tree(node, pos=None, level=0, width=2., vert_gap=0.2, xcenter=0.5):
    """Calculate positions for drawing a tree iteratively"""
    if pos is None:
        pos = {}
    
    # Use a queue for breadth-first traversal
    queue = deque([(node, xcenter, 0, width)])
    
    while queue:
        current_node, x, curr_level, curr_width = queue.popleft()
        pos[current_node.name] = (x, 1 - curr_level * vert_gap)
        
        if current_node.children:
            child_count = len(current_node.children)
            dx = curr_width / child_count
            start_x = x - curr_width/2 + dx/2
            
            for i, child in enumerate(current_node.children):
                child_x = start_x + i * dx
                queue.append((child, child_x, curr_level + 1, dx))
    
    return pos

def draw_tree_to_pdf(tree, outfile):
    """Draw tree structure to a PDF file iteratively"""
    graph = nx.DiGraph()
    
    # Build graph iteratively
    queue = deque([tree])
    node_info = {}
    
    while queue:
        node = queue.popleft()
        graph.add_node(node.name)
        node_info[node.name] = node
        
        for child in node.children:
            graph.add_edge(node.name, child.name)
            queue.append(child)
    
    pos = draw_tree(tree)
    
    # Prepare node labels
    node_labels = {}
    for node_name in graph.nodes():
        node = node_info[node_name]
        if node.cell_no is not None:
            node_labels[node_name] = f"{node_name}\n({node.cell_no} cells)"
        else:
            node_labels[node_name] = node_name
    
    plt.figure(figsize=(20, 10))
    nx.draw_networkx_edges(graph, pos, arrows=False)
    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color="skyblue")
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=8, font_color="black")
    
    plt.savefig(outfile, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()

def cal_tree_depth(tree):
    """Calculate the depth of a tree iteratively"""
    if not tree:
        return 0
    
    max_depth = 0
    queue = deque([(tree, 0)])
    
    while queue:
        node, depth = queue.popleft()
        max_depth = max(max_depth, depth)
        
        for child in node.children:
            queue.append((child, depth + 1))
    
    return max_depth

def count_nodes(tree):
    """Count the total number of nodes in the tree iteratively"""
    if not tree:
        return 0
    
    count = 0
    queue = deque([tree])
    
    while queue:
        node = queue.popleft()
        count += 1
        queue.extend(node.children)
    
    return count

def update_depths(root):
    """Update depth values for all nodes iteratively"""
    if not root:
        return
    queue = deque([(root, 0)])
    
    while queue:
        node, depth = queue.popleft()
        node.depth = depth
        
        for child in node.children:
            queue.append((child, depth + 1))

def collect_all_nodes(root, mode=1):
    """Collect all nodes iteratively"""
    if not root:
        return []
    
    all_nodes = []
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        
        if mode == 0 and node.name == 'normal':
            pass
        else:
            all_nodes.append(node)
        
        queue.extend(node.children)
    
    return all_nodes

def generate_beta_splits(n_splits, alpha, beta):
    """Generate split points using Beta distribution"""
    return np.random.beta(alpha, beta, n_splits)

def determine_num_children(max_children, geometric_p=0.5):
    """Determine number of children for a node"""
    if max_children <= 2:
        return 2
    
    # Geometric distribution favors smaller values
    num = 2 + np.random.geometric(geometric_p) - 1 # P(k) = (1-p)^(k-1) * p for k=1,2,...
    return min(num, max_children)

# --- MODIFICATION START ---
# Simplified build_tree_beta_splitting to focus only on generating clones.
def build_tree_beta_splitting(num_clones, alpha, beta, max_children=3):
    """
    Build a tree using the Beta Splitting Model - completely iterative approach.
    This function's sole responsibility is to generate a tree with the target
    number of clones. It will always return a tree.

    Parameters:
    - num_clones: Target number of clones (excluding root)
    - alpha: Beta distribution alpha parameter
    - beta: Beta distribution beta parameter
    - max_children: Maximum children per node
    
    Returns:
    - Root node of the constructed tree
    """
    # Handle edge case of zero clones
    if num_clones == 0:
        root = TreeNode("normal")
        root.interval = [0.0, 1.0]
        root.id = 0
        root.depth = 0
        return root

    # Initialize root
    root = TreeNode("normal")
    root.interval = [0.0, 1.0]
    root.id = 0
    root.depth = 0
    
    # Generate branch lengths
    n_edges = num_clones * 2
    branch_lengths = np.random.exponential(1.0, n_edges)
    total_length = np.sum(branch_lengths)
    branch_lengths = branch_lengths / total_length if total_length > 0 else np.zeros(n_edges)
    
    # Track leaves and all nodes
    leaves = [root]
    all_nodes = [root]
    node_id = 0
    edge_idx = 0
    
    # Generate tree iteratively until the desired number of clones is reached
    while node_id < num_clones:
        if not leaves:
            # This case should ideally not be reached if num_children >= 2,
            # but as a safeguard, we break and let post-processing handle it.
            break

        # Select a leaf to split
        # Use Beta distribution to select which leaf
        if len(leaves) == 1:
            leaf_idx = 0
        else:
            selection_prob = np.random.beta(alpha, beta)
            leaf_idx = int(selection_prob * len(leaves))
            leaf_idx = min(leaf_idx, len(leaves) - 1)
        
        parent = leaves.pop(leaf_idx)
        
        # Determine number of children
        remaining_to_create = num_clones - node_id
        # We need to create at least one child to make progress.
        # The parent node will be replaced by its children in the leaves list.
        # If we create C children, we use up 1 leaf and add C leaves. Net change: C-1.
        # We also create C new clones.
        
        # Decide how many children this node will have
        num_children = determine_num_children(max_children)
        # Ensure we don't create vastly more clones than needed.
        # If we need 3 clones, and num_children is 4, we should cap it at 3.
        num_children = min(num_children, remaining_to_create)
        # We must have at least 2 children to split, unless we are at the very end
        # and only one clone is left to be created. But even then, a split implies >= 2 branches.
        # Let's enforce at least 2 children, unless that would exceed the total clone count.
        if remaining_to_create > 1:
            num_children = max(2, num_children)
        else: # remaining_to_create is 1
            num_children = 2 # Create two children, but we will only label one as a new clone ID
                             # This is a modeling choice. A simpler way is to just create one.
                             # Let's stick to the logic that a split creates >= 2 children.
                             # The loop condition `while node_id < num_clones` will handle termination.
        
        # Generate split points
        if num_children == 2:
            split_points = [np.random.beta(alpha, beta)]
        else:
            raw_splits = generate_beta_splits(num_children - 1, alpha, beta)
            split_points = sorted(raw_splits)
        
        # Create children based on split points
        start, end = parent.interval
        interval_size = end - start
        
        abs_splits = [start + sp * interval_size for sp in split_points]
        boundaries = [start] + abs_splits + [end]
        
        # Create child nodes
        for i in range(num_children):
            if node_id >= num_clones:
                break

            node_id += 1
            child = TreeNode(f"clone{node_id}")
            child.interval = [boundaries[i], boundaries[i + 1]]
            child.parent = parent
            child.id = node_id
            
            # Assign branch length
            if edge_idx < len(branch_lengths):
                child.edge_length = branch_lengths[edge_idx]
                edge_idx += 1
            else:
                child.edge_length = np.random.exponential(0.1) # Assign a small random length if we run out
            
            parent.children.append(child)
            all_nodes.append(child)
            leaves.append(child)
    
    # Update all depths once at the end
    update_depths(root)
    
    return root

def balance_tree(root, balance_factor=0.8):
    """
    Balance the tree by redistributing nodes across depths.
    This version is less aggressive and avoids excessive pruning.
    
    Parameters:
    - root: Root of the tree
    - balance_factor: How strongly to enforce balance (0-1)
    """
    if balance_factor <= 0 or not root.children:
        return
    
    # Collect all nodes and their depths
    all_nodes = collect_all_nodes(root, mode=1)
    
    if len(all_nodes) <= 2:
        return
    
    # Calculate depth statistics
    depths = [node.depth for node in all_nodes if node.depth is not None]
    if not depths: return

    avg_depth = np.mean(depths)
    std_depth = np.std(depths)
    
    # Identify nodes that are too deep
    # A high balance_factor means less pruning
    threshold = avg_depth + std_depth * (1 / (balance_factor + 0.01))
    
    nodes_to_prune = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        
        # A node is a candidate for pruning if it's a leaf, too deep, and its parent has other children
        if not node.children and node.depth > threshold and node.parent and len(node.parent.children) > 1:
            nodes_to_prune.append(node)
        else:
            queue.extend(node.children)
    
    for node in nodes_to_prune:
        if node.parent:
            node.parent.children.remove(node)

    # Update depths after pruning
    update_depths(root)

# def assign_cells_to_all_nodes(root, cell_num):
#     """
#     Assign cells to all nodes of the tree, not just leaves.
    
#     Parameters:
#     - root: Root node of the tree
#     - cell_num: Total number of cells to distribute
    
#     Returns:
#     - Updated tree with cell_no assigned to all nodes
#     """
#     # Collect all nodes in the tree iteratively
#     all_nodes = []
#     queue = [root]
    
#     while queue:
#         node = queue.pop(0)
#         all_nodes.append(node)
#         queue.extend(node.children)
    
#     # Assign cells proportionally to all nodes, with more cells to deeper nodes
#     # Calculate weighted distribution based on depth
#     total_weight = 0
#     weights = []
    
#     for node in all_nodes:
#         # Give higher weight to deeper nodes (more evolved clones)
#         weight = 1.0 + node.depth * 0.5
#         weights.append(weight)
#         total_weight += weight
    
#     # Distribute cells based on weights
#     remaining_cells = cell_num
#     for i, node in enumerate(all_nodes):
#         # Calculate cell allocation proportionally
#         node_cells = int((weights[i] / total_weight) * cell_num)
#         node.cell_no = node_cells
#         remaining_cells -= node_cells
    
#     # Distribute any remaining cells due to rounding
#     for i in range(remaining_cells):
#         all_nodes[i % len(all_nodes)].cell_no += 1
    
#     return root

def assign_cells_to_all_nodes(root, cell_num, mode=2):
    """
    Assign cells to all nodes of the tree based on the selected distribution mode.
    
    Parameters:
    - root: Root node of the tree
    - cell_num: Total number of cells to distribute
    - mode: Distribution strategy (int)
        0: Uniform distribution (all nodes get roughly equal cells)
        1: Depth-decreasing (deeper nodes get fewer cells - ancestral dominance)
        2: Depth-increasing (deeper nodes get more cells - evolved dominance)
    
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
    
    # Calculate weights based on the selected mode
    weights = []
    gamma = 0.5 # Scaling factor
    
    for node in all_nodes:
        if mode == 0:
            # Uniform distribution
            weight = 1.0
        elif mode == 1:
            # Deeper nodes get fewer cells (Ancestral Dominance)
            # Using inverse relationship
            weight = 1.0 / (1.0 + node.depth * gamma)
        elif mode == 2:
            # Deeper nodes get more cells (Evolved Dominance)
            # Original logic
            weight = 1.0 + node.depth * gamma
        else:
            raise ValueError("Mode must be 0, 1, or 2")
            
        weights.append(weight)
    
    total_weight = sum(weights)
    
    # Distribute cells based on weights
    remaining_cells = cell_num
    for i, node in enumerate(all_nodes):
        # Calculate cell allocation proportionally
        if total_weight > 0:
            node_cells = int((weights[i] / total_weight) * cell_num)
        else:
            node_cells = 0 # Should not happen unless weights are 0
            
        node.cell_no = node_cells
        remaining_cells -= node_cells
    
    # Distribute any remaining cells due to rounding
    # (Simple round-robin to the first few nodes)
    for i in range(remaining_cells):
        all_nodes[i % len(all_nodes)].cell_no += 1
    
    return root

def generate_tree_beta(cell_num=1000, num_clones=10, alpha=10.0, beta=10.0, 
                      treedepth=4, treedepthsigma=0.5, max_children=3, 
                      balance_factor=0.8, mode=0, seed=None):
    """
    Generate a random tree using the Beta Splitting Model, ensuring a tree is always produced.
    This function is robust and avoids recursion errors.

    Parameters are the same as before.
    
    Returns:
    - Root node of the generated tree
    """
    # Set random seed for reproducibility
    if seed is not None:
        set_random_seed(seed)
    
    max_attempts = 20
    best_tree = None
    best_depth_diff = float('inf')
    
    for attempt in range(max_attempts):
        try:
            # Step 1: Generate a tree with the exact number of clones.
            # This function is now guaranteed to return a tree.
            root = build_tree_beta_splitting(
                num_clones=num_clones,
                alpha=alpha,
                beta=beta,
                max_children=max_children
            )
            
            # Step 2: Apply balancing if requested.
            # Note: Balancing might prune nodes and reduce the clone count.
            if balance_factor > 0:
                balance_tree(root, balance_factor)
            
            # Step 3: Check if the tree is still valid (e.g., has enough clones).
            final_clones = collect_all_nodes(root, mode=0)
            if len(final_clones) < num_clones * 0.5: # Allow for some pruning, but not too much
                continue
            
            # Step 4: Evaluate the tree based on depth.
            actual_depth = cal_tree_depth(root)
            depth_diff = abs(actual_depth - treedepth)
            
            # If this is the best tree so far (closest to target depth), save it.
            if depth_diff < best_depth_diff:
                best_depth_diff = depth_diff
                best_tree = root

            # If we find a "perfect" tree, we can stop early.
            if depth_diff <= treedepthsigma:
                break
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue
    
    # After all attempts, if best_tree is still None, it means all attempts failed validation.
    # We must generate a fallback tree.
    if best_tree is None:
        best_tree = build_tree_beta_splitting(
            num_clones=num_clones,
            alpha=alpha,
            beta=beta,
            max_children=max_children
        )
        # Avoid balancing on the fallback tree to ensure clone count.
    
    # Final Step: Assign cells to the chosen tree.
    assign_cells_to_all_nodes(best_tree, cell_num, mode=mode)
    
    return best_tree

def load_tree_from_newick(newick_string, cell_num=1000):
    """
    Load a tree from Newick format string and prepare it for use.
    Performs the same post-processing as generate_tree_beta:
    - Updates depths
    - Assigns cells proportionally
    
    Parameters:
    - newick_string: Newick format string (e.g., "((clone4)clone1,...)normal;")
    - cell_num: Total number of cells to distribute
    
    Returns:
    - Root node of the loaded tree
    """
    
    def parse_newick_iterative(newick_str):
        """Parse Newick format string iteratively to build tree"""
        # Remove trailing semicolon and whitespace
        newick_str = newick_str.strip().rstrip(';').strip()
        
        if not newick_str:
            raise ValueError("Empty Newick string")
        
        # Stack to track current path: (node, start_position)
        stack = []
        root = None
        current_node = None
        i = 0
        
        while i < len(newick_str):
            char = newick_str[i]
            
            if char == '(':
                # Start a new internal node (parent)
                # We'll set its name later when we encounter it after ')'
                new_node = TreeNode("")  # Placeholder name
                
                if current_node is not None:
                    new_node.parent = current_node
                    current_node.children.append(new_node)
                
                stack.append(current_node)
                current_node = new_node
                
                if root is None:
                    root = new_node
                
                i += 1
                
            elif char == ')':
                # End of children list, next comes the parent's name
                if not stack:
                    raise ValueError("Unbalanced parentheses in Newick string")
                
                # Move back to parent
                parent = stack.pop()
                
                # Read the node name and optional branch length
                i += 1
                name_start = i
                while i < len(newick_str) and newick_str[i] not in '(),;:':
                    i += 1
                
                name = newick_str[name_start:i].strip()
                if name:
                    current_node.name = name
                else:
                    # If no name provided, generate one
                    current_node.name = f"internal_{id(current_node)}"
                
                # Check for branch length
                if i < len(newick_str) and newick_str[i] == ':':
                    i += 1
                    length_start = i
                    while i < len(newick_str) and newick_str[i] not in '(),;':
                        i += 1
                    try:
                        current_node.edge_length = float(newick_str[length_start:i])
                    except ValueError:
                        current_node.edge_length = 0.0
                else:
                    current_node.edge_length = 0.0
                
                current_node = parent
                
            elif char == ',':
                # Sibling separator
                i += 1
                
            elif char in ' \t\n\r':
                # Skip whitespace
                i += 1
                
            else:
                # Read a leaf node name
                name_start = i
                while i < len(newick_str) and newick_str[i] not in '(),;:':
                    i += 1
                
                name = newick_str[name_start:i].strip()
                if not name:
                    raise ValueError("Empty node name encountered")
                
                leaf_node = TreeNode(name)
                
                # Check for branch length
                if i < len(newick_str) and newick_str[i] == ':':
                    i += 1
                    length_start = i
                    while i < len(newick_str) and newick_str[i] not in '(),;':
                        i += 1
                    try:
                        leaf_node.edge_length = float(newick_str[length_start:i])
                    except ValueError:
                        leaf_node.edge_length = 0.0
                else:
                    leaf_node.edge_length = 0.0
                
                if current_node is not None:
                    leaf_node.parent = current_node
                    current_node.children.append(leaf_node)
                else:
                    # This is a single-node tree
                    root = leaf_node
                    current_node = leaf_node
        
        if not root:
            raise ValueError("Failed to parse Newick string")
        
        return root
    
    # Parse the Newick string
    root = parse_newick_iterative(newick_string)
    
    # Assign intervals based on tree structure
    # Use a simple approach: divide [0,1] among siblings proportionally
    def assign_intervals(node, start=0.0, end=1.0):
        """Assign intervals to nodes based on their position in the tree"""
        node.interval = [start, end]
        
        if not node.children:
            return
        
        # Divide the interval equally among children
        interval_size = end - start
        child_count = len(node.children)
        child_width = interval_size / child_count
        
        for i, child in enumerate(node.children):
            child_start = start + i * child_width
            child_end = child_start + child_width
            assign_intervals(child, child_start, child_end)
    
    assign_intervals(root)
    
    # Assign node IDs
    node_id = 0
    queue = deque([root])
    while queue:
        node = queue.popleft()
        node.id = node_id
        node_id += 1
        queue.extend(node.children)
    
    # Update depths for all nodes
    update_depths(root)
    
    # Assign cells proportionally to all clone nodes
    assign_cells_to_all_nodes(root, cell_num)
    
    return root

def save_tree_to_file(root, filename):
    """Save the tree to a file in JSON format."""
    def convert_numpy(obj):
        """Convert numpy types to Python types"""
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    tree_dict = root.to_dict()
    tree_dict = convert_numpy(tree_dict)
    
    with open(filename, "w") as file:
        json.dump(tree_dict, file, indent=4)

def load_tree_from_file(filename):
    """Load the tree from a JSON file."""
    with open(filename, "r") as file:
        data = json.load(file)
        return TreeNode.from_dict(data)

def tree_to_newick(node):
    """Convert tree to Newick format iteratively"""
    # Build tree representation bottom-up
    node_strings = {}
    
    # Process nodes in post-order (children before parents)
    stack = [(node, False)]
    processed = set()
    
    while stack:
        current, visited = stack.pop()
        
        if visited or not current.children:
            # Process this node
            if not current.children:
                result = current.name
            else:
                child_strs = [node_strings[child.name] for child in current.children if child.name in node_strings]
                result = f"({','.join(child_strs)}){current.name}"
            
            # Add cell no
            result += f":{current.cell_no}cells"

            node_strings[current.name] = result
            processed.add(current.name)
        else:
            # Push current node back to process after children
            stack.append((current, True))
            # Push children
            for child in reversed(current.children):
                if child.name not in processed:
                    stack.append((child, False))
    
    # The Newick format for the root itself might not include its own name if it's the ancestor
    # Let's adjust for the standard where the root might be implicit or explicit.
    final_str = node_strings[node.name]
    if not node.children:
        return f"{node.name}:{node.cell_no}cells;"
    
    # For a tree, the final string is just the root's representation + semicolon
    return final_str + ";"

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
            node.children = new_node.children
            node.maternal_cnas = new_node.maternal_cnas
            node.paternal_cnas = new_node.paternal_cnas
            node.maternal_fasta = new_node.maternal_fasta
            node.paternal_fasta = new_node.paternal_fasta
            node.fasta = new_node.fasta
            node.maternal_fasta_length = new_node.maternal_fasta_length
            node.paternal_fasta_length = new_node.paternal_fasta_length
            node.parent = new_node.parent
            node.ratio = new_node.ratio
            node.cell_no = new_node.cell_no
            node.depth = new_node.depth
            node.changes = new_node.changes
            node.cna_status = new_node.cna_status

            return True
        for child in node.children:
            if dfs(child):
                return True
        return False

    dfs(root)
    return root

def main():
    """Main function to parse arguments and generate the tree"""
    parser = argparse.ArgumentParser(
        description='Generate phylogenetic trees using the Beta Splitting Model'
    )
    
    parser.add_argument('-c', '--cell-num', type=int, default=10000,
                        help='Total number of cells (default: 10000)')
    parser.add_argument('-n', '--num-clones', type=int, default=10,
                        help='Number of clones excluding normal (default: 10)')
    parser.add_argument('-B', '--Beta', type=float, default=10.0,
                        help='Beta parameter (higher = more balanced) (default: 10.0)')
    parser.add_argument('-A', '--Alpha', type=float, default=10.0,
                        help='Alpha parameter (higher = more balanced) (default: 10.0)')
    parser.add_argument('-G', '--treedepth', type=float, default=4,
                        help='Target tree depth (default: 4)')
    parser.add_argument('-K', '--treedepthsigma', type=float, default=0.5,
                        help='Tree depth tolerance (default: 0.5)')
    parser.add_argument('-M', '--max-children', type=int, default=3,
                        help='Maximum children per node (default: 3)')
    parser.add_argument('-b', '--balance', type=float, default=0.8,
                        help='Balance factor 0-1 (0=no balance) (default: 0.8)')
    parser.add_argument('-o', '--output', type=str, default='tree',
                        help='Output file prefix (default: "tree")')
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    actual_seed = set_random_seed(args.seed)
    
    # Generate tree
    root = generate_tree_beta(
        cell_num=args.cell_num,
        num_clones=args.num_clones,
        alpha=args.Alpha,
        beta=args.Beta,
        treedepth=args.treedepth,
        treedepthsigma=args.treedepthsigma,
        max_children=args.max_children,
        balance_factor=args.balance,
        seed=actual_seed
    )
    
    # Save outputs
    save_tree_to_file(root, f"{args.output}.json")
    draw_tree_to_pdf(root, f"{args.output}.pdf")
    
    # Generate statistics
    all_nodes = collect_all_nodes(root, mode=1)
    clone_nodes = collect_all_nodes(root, mode=0)
    leaves = [n for n in all_nodes if not n.children]
    
    # Depth distribution
    depths = {}
    for node in all_nodes:
        d = node.depth if node.depth is not None else 0
        depths[d] = depths.get(d, 0) + 1
    
    # Save Newick format
    with open(f"{args.output}.newick", "w") as f:
        f.write(tree_to_newick(root))
    
    # Save metadata
    with open(f"{args.output}_metadata.txt", "w") as f:
        f.write(f"Random seed: {actual_seed}\n")
        f.write(f"Alpha: {args.Alpha}\n")
        f.write(f"Beta: {args.Beta}\n")
        f.write(f"Target number of clones: {args.num_clones}\n")
        f.write(f"Actual number of clones: {len(clone_nodes)}\n")
        f.write(f"Balance factor: {args.balance}\n")
        f.write(f"Target tree depth: {args.treedepth}\n")
        f.write(f"Actual tree depth: {cal_tree_depth(root)}\n")
        f.write(f"Leaf nodes: {len(leaves)}\n")
        f.write(f"Total nodes: {len(all_nodes)}\n")
        f.write(f"Max children: {args.max_children}\n")
    
    print(f"\nFiles saved:")
    print(f"  - {args.output}.json")
    print(f"  - {args.output}.pdf")
    print(f"  - {args.output}.newick")
    print(f"  - {args.output}_metadata.txt")
    print(f"\nTo reproduce this exact tree, use the command-line argument: -s {actual_seed}")
    
    return root

if __name__ == "__main__":
    main()