#!/usr/bin/env python3
import torch
import argparse
from torch_geometric.data import Data
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd




def load_data(node_file, edge_file):
    """
    Load nodes and edges from CSV files.
    """
      # Load the node features from CSV
    node_features_df = pd.read_csv(node_file)
    node_ids = node_features_df['id']
    #this is the part of the code that will not currently work as need to parse the data
   # node_features = torch.tensor(node_features_df.drop(columns=['id']).values, dtype=torch.float)
    # Generate random node features as a place holder because node feature parsing to be added
   # node_features = torch.rand((num_nodes, 2))  # 2D features for each node



     
    print(node_features_df.columns)

    # Define columns that contain multi-value entries separated by 'ǂ'
    multi_value_columns = ['all_names', 'all_categories', 'equivalent_curies', 'publications', 'label']

    # Split multi-value fields
    for col in multi_value_columns:
        node_features_df[col] = node_features_df[col].apply(lambda x: x.split('ǂ') if isinstance(x, str) else x)



    # Extract the 'equivalent_curies' column for one-hot encoding
    curies = node_features_df['equivalent_curies'].explode().unique()  # Get all unique CURIEs
    curies = [curie.strip() for curie in curies]  # Clean up extra spaces, if any

    # One-hot encode the 'equivalent_curies' for each node
    mlb = MultiLabelBinarizer()
    one_hot_curies = mlb.fit_transform(node_features_df['equivalent_curies'])  # Each row gets a binary vector

    # Convert the one-hot encoded matrix to a PyTorch tensor
    node_features = torch.tensor(one_hot_curies, dtype=torch.float)


    # Load the edge data from CSV
    edge_data_df = pd.read_csv(edges_file)
    edge_index = torch.tensor(edge_data_df[['source', 'target']].values.T, dtype=torch.long)

    # Extract unique edge types from the 'type' column
    edge_types = edge_data_df['predicate'].values
    unique_edge_types = sorted(set(edge_types))  # Sort to ensure consistent mapping

    # Create a mapping of edge types to integer values
    edge_type_mapping = {etype: idx for idx, etype in enumerate(unique_edge_types)}

    # Convert edge types to their corresponding numeric values
    edge_attr = torch.tensor([edge_type_mapping[et] for et in edge_types], dtype=torch.long)

 
    
    return node_features, node_ids, edge_index, edge_attr

 

def build_knowledge_graph(node_file, edge_file, output_file):
    """
    Build a knowledge graph from node and edge CSV files and save it to a .pt file.
    """
    node_features, node_ids, edge_index, edge_attr = load_data(node_file, edge_file)
       # Create a torch_geometric Data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    # Save the graph object to a file
    torch.save(data, output_file)
    print(f"Graph saved to {output_file}")
  
    # Save node IDs separately when saving the graph

    node_ids.to_csv("node_ids.csv", index=False)
 
    
def main():
    parser = argparse.ArgumentParser(description="Build graph")
    
    # Input data files
    parser.add_argument('--node_file', required=True, help='Path to the node CSV file')
    parser.add_argument('--edge_file', required=True, help='Path to the edge CSV file')
    
    # Output model file
    parser.add_argument('--output_file', required=True, help='Path to save the trained model ')

    # Parse arguments
    args = parser.parse_args()
    
    # Call the train_ml_model function
    build_knowledge_graph(args.node_file, args.edge_file, args.output_file)

if __name__ == '__main__':
    main()



  
