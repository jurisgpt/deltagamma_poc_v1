#!/usr/bin/env python3
"""
Build a biological knowledge graph from CSV files of nodes and edges.
"""
import os
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from torch_geometric.data import Data


def load_data(node_file, edge_file):
    """
    Load nodes and edges from CSV files and prepare feature and adjacency data.

    Parameters:
    - node_file: Path to a node CSV containing an 'id' column and multi-valued fields.
    - edge_file: Path to an edge CSV containing 'source', 'target', and 'predicate' columns.

    Returns:
    - node_features (Tensor): One-hot encoded features for each node.
    - node_ids (Series): Original node IDs in the same order as features.
    - edge_index (LongTensor): Source/target index pairs for graph edges.
    - edge_attr (LongTensor): Encoded edge predicate types per edge.
    """
    # Load node table
    node_df = pd.read_csv(node_file)
    node_ids = node_df['id']

    # Parse multi-valued fields separated by 'ǂ'
    multi_cols = ['all_names', 'all_categories', 'equivalent_curies', 'publications', 'label']
    for col in multi_cols:
        if col in node_df.columns:
            node_df[col] = node_df[col].apply(lambda x: x.split('ǂ') if isinstance(x, str) else [])

    # One-hot encode the 'equivalent_curies' field as node features
    if 'equivalent_curies' in node_df.columns:
        mlb = MultiLabelBinarizer()
        one_hot = mlb.fit_transform(node_df['equivalent_curies'])
    else:
        # Fallback to zero-dimension features if missing
        one_hot = np.zeros((len(node_df), 0), dtype=int)
    node_features = torch.tensor(one_hot, dtype=torch.float)

    # Load edge table
    edge_df = pd.read_csv(edge_file)
    edge_index = torch.tensor(edge_df[['source', 'target']].values.T, dtype=torch.long)

    # Encode edge predicate types as integers
    if 'predicate' in edge_df.columns:
        predicates = edge_df['predicate']
    else:
        predicates = edge_df.iloc[:, 2]
    unique_preds = sorted(predicates.unique())
    pred2idx = {p: i for i, p in enumerate(unique_preds)}
    edge_attr = torch.tensor([pred2idx[p] for p in predicates], dtype=torch.long)

    return node_features, node_ids, edge_index, edge_attr


def build_knowledge_graph(node_file, edge_file, output_file, node_ids_file='node_ids.csv'):
    """
    Build and save a knowledge graph Data object and corresponding node ID lookup.

    Parameters:
    - node_file: Path to the node CSV file.
    - edge_file: Path to the edge CSV file.
    - output_file: Path to save the PyG Data object (.pt).
    - node_ids_file: Path to save the node ID lookup CSV.
    """
    x, ids, edge_index, edge_attr = load_data(node_file, edge_file)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    torch.save(data, output_file)
    print(f"Graph saved to {output_file}")
    # Save node IDs to preserve mapping from index to original ID
    ids.to_csv(node_ids_file, index=False)
    print(f"Node IDs saved to {node_ids_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Build a knowledge graph from node and edge CSV files"
    )
    parser.add_argument('--node_file', required=True,
                        help='Path to the node CSV file')
    parser.add_argument('--edge_file', required=True,
                        help='Path to the edge CSV file')
    parser.add_argument('--output_file', required=True,
                        help='Path to save the graph object (.pt)')
    parser.add_argument('--node_ids_file', default='node_ids.csv',
                        help='Path to save the node IDs CSV')
    args = parser.parse_args()
    build_knowledge_graph(
        args.node_file, args.edge_file,
        args.output_file, args.node_ids_file
    )


if __name__ == '__main__':
    main()