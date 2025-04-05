import matplotlib.pyplot as plt
import networkx as nx
import random
from collections import deque
import json

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
            "changes": self.changes
        }

    @staticmethod
    def from_dict(data, parent=None):
        """Restore a TreeNode from a dictionary, setting parent references."""
        node = TreeNode(data["name"])
        node.maternal_cnvs = data["maternal_cnvs"]
        node.paternal_cnvs = data["paternal_cnvs"]
        node.maternal_fasta = data["maternal_fasta"]
        node.paternal_fasta = data["paternal_fasta"]
        node.fq1 = data["fq1"]
        node.fq2 = data["fq2"]
        node.fasta = data["fasta"]
        node.maternal_fasta_length = data["maternal_fasta_length"]
        node.paternal_fasta_length = data["paternal_fasta_length"]
        node.ratio = data["ratio"]
        node.cell_no = data["cell_no"]
        node.depth = data["depth"]
        node.changes = data["changes"]
        node.parent = parent
        node.children = [TreeNode.from_dict(child, node) for child in data["children"]]
        return node
    
def draw_tree(node, pos=None, level=0, width=2., vert_gap=0.2, xcenter=0.5):
    if pos is None:
        pos = {node.name: (xcenter, 1 - level * vert_gap)}
    else:
        pos[node.name] = (xcenter, 1 - level * vert_gap)
    if node.children:
        dx = width / 2
        nextx = xcenter - width / 2 - dx / 2
        for child in node.children:
            nextx += dx
            pos = draw_tree(child, pos=pos,
                            level=level + 1,
                            width=dx, xcenter=nextx)
    return pos

def generate_random_tree(node_count):
    nodes = [TreeNode(i) for i in range(1, node_count + 1)]
    used_nodes = []
    root = random.choice(nodes)
    nodes.remove(root)
    used_nodes.append(root)
    while nodes:
        node = random.choice(nodes)
        nodes.remove(node)
        parent = random.choice(used_nodes)
        parent.children.append(node)
        node.parent = parent
        used_nodes.append(node)
    
    return root

def cal_tree_depth(tree):
    if len(tree.children) == 0:
        return 1

    return max(cal_tree_depth(child) for child in tree.children) + 1

# def draw_tree_to_pdf(random_tree, outfile):
#     # Creating a graph for drawing
#     graph = nx.DiGraph()
#     graph.add_node(random_tree.name)
#     def add_edges(node):
#         for child in node.children:
#             graph.add_edge(node.name, child.name)
#             add_edges(child)
#     add_edges(random_tree)

#     # Drawing the tree using matplotlib
#     pos = draw_tree(random_tree)
#     plt.figure(figsize=(20, 10))
#     nx.draw(graph, pos, with_labels=True, arrows=False, node_size=700, node_color="skyblue", font_size=8, font_color="black")

#     # Save the tree graph to a PDF file
#     plt.savefig(outfile, format="pdf", dpi=300, bbox_inches="tight")
def draw_tree_to_pdf(random_tree, outfile):
    # Creating a graph for drawing
    graph = nx.DiGraph()
    graph.add_node(random_tree.name)
    def add_edges(node):
        for child in node.children:
            graph.add_edge(node.name, child.name)
            add_edges(child)
    add_edges(random_tree)

    # Calculating positions of nodes
    pos = draw_tree(random_tree)

    # Drawing the tree using matplotlib
    plt.figure(figsize=(20, 10))

    # Adjust horizontal spacing based on the number of children
    max_children = max(len(child.children) for child in random_tree.children)
    horizontal_spacing = 1 / (max_children + 1)

    # Draw the tree with adjusted positions
    nx.draw(graph, pos, with_labels=True, arrows=False, node_size=700, node_color="skyblue", font_size=8, font_color="black")
    
    # Save the tree graph to a PDF file
    plt.savefig(outfile, format="pdf", dpi=300, bbox_inches="tight")


def rename_tree_nodes(root, start_value=0):
    """
    Rename tree nodes starting from the given start_value based on depth.
    
    Parameters:
    - root: The root node of the tree.
    - start_value: The starting value for renaming nodes (default is 0).
    """
    queue = deque([(root, 0)])
    current_value = start_value

    while queue:
        node, depth = queue.popleft()

        # Rename the current node
        if depth == 0:
            node.name = 'normal'
        else:
            node.name = 'clone' + str(current_value)
        node.depth = depth
        current_value += 1

        # Add children to the stack with increased depth
        queue.extend((child, depth + 1) for child in node.children)

def generate_random_tree_balance(node_count, max_depth):
    # Example usage
    tree_depth = 1000
    while tree_depth > max_depth:
        root = generate_random_tree(node_count)
        tree_depth = cal_tree_depth(root)

    # rename node name
    rename_tree_nodes(root)
    return root

def tree_to_newick(node):
    if not node.children:
        # If the node has no children, return its name
        return node.name
    else:
        # If the node has children, recursively process each child
        child_strings = [tree_to_newick(child) for child in node.children]

        # Join child strings with commas and enclose in parentheses
        children_str = ",".join(child_strings)
        result = f"({children_str})"

        # Append the current node's name
        if node.name:
            result += node.name

        # You can add additional information if needed
        # For example, adding maternal and paternal CNVs
        # if node.maternal_cnvs:
        #     result += f"[maternal_cnvs={','.join(map(str, node.maternal_cnvs))}]"
        # if node.paternal_cnvs:
        #     result += f"[paternal_cnvs={','.join(map(str, node.paternal_cnvs))}]"
        if node.ratio:
            result += "[ratio={0}]".format(node.ratio)
        if node.cell_no:
            result += "[cell_no={0}]".format(node.cell_no)

        return result

# tree = generate_random_tree_balance(10, 4)
# draw_tree_to_pdf(tree, 'tree_graph.pdf')

def save_tree_to_file(root, filename):
    """Save the tree to a file in JSON format."""
    with open(filename, "w") as file:
        json.dump(root.to_dict(), file, indent=4)

def load_tree_from_file(filename):
    """Load the tree from a JSON file."""
    with open(filename, "r") as file:
        data = json.load(file)
        return TreeNode.from_dict(data)
    
def collect_all_nodes(root, mode=0):
    """
    Collect all nodes of a tree into an array.
    :param root: The root node of the tree.
    :return: A list containing all nodes in the tree.
    """
    all_nodes = []

    def dfs(node):
        if node is None:
            return
        if mode == 0: # remove normal node
            if node.name != 'normal':
                all_nodes.append(node)  # Add the current node to the array
        else:
            all_nodes.append(node)
        for child in node.children:
            dfs(child)  # Recursively visit children

    dfs(root)
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