import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder  # To encode edge types as numbers

def generate_random_graph(num_nodes=100, num_edges=500,num_test=200,output_file="Output_Graph"):
    """
    Generate a random graph with node features, edge types, and export them to files.
    
    Args:
    - num_nodes: The number of nodes in the graph.
    - num_edges: The number of edges in the graph.
    - num_test: The number of pairs to generate random classification labels for
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate a random graph
    # Edge index: 2 x num_edges tensor (connects nodes)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)

    # Generate random node features 
    node_features = torch.rand((num_nodes, 2))  # 2D features for each node

    # Generate random edge types (3 types for each edge)
    edge_types = np.random.choice(['type_1', 'type_2', 'type_3'], size=num_edges)

    # Generate random labels for use in training
    class_index = torch.randint(0, num_nodes, (2, num_test), dtype=torch.long)
    # Generate random label classification
    class_y = torch.randint(0,2,(1,num_test),dtype=torch.long)

    # Convert edge types from strings to integers using LabelEncoder
    label_encoder = LabelEncoder()
    edge_types_encoded = label_encoder.fit_transform(edge_types)  # Encode to integers

    #  Create the PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=edge_index)

    # Assign encoded edge types to edge_attr
    data.edge_attr = torch.tensor(edge_types_encoded, dtype=torch.long)

    # Save the graph object to a file
    torch.save(data, output_file)
    print(f"Graph saved to {output_file}")
    #  Export to .csv (edges and edge types)
    edges_df = pd.DataFrame({
        'source': edge_index[0].numpy(),
        'target': edge_index[1].numpy(),
        'type': edge_types  # Keep the original string edge types for export
    })
    edges_df.to_csv('edges.csv', index=False)  # Save edges and edge types to .csv file
    
    #  Export to .csv (edges and edge types)
    class_df = pd.DataFrame({
        'source': class_index[0].numpy(),
        'target': class_index[1].numpy(),
        'y': class_y[0].numpy() 
    })
    class_df.to_csv('Class_Labels.csv', index=False) 
    
    # Export the node features along with their IDs to .csv
    node_features_df = pd.DataFrame(node_features.numpy(), columns=['feature_1', 'feature_2'])
    node_features_df.insert(0, 'id', np.arange(num_nodes))  # Add a column for node IDs
    node_features_df.to_csv('node_features.csv', index=False)  # Save node features with IDs to .csv
    # Save node IDs separately when saving the graph
    node_ids = node_features_df['id']

    node_ids.to_csv("node_ids.csv", index=False)

def main():
    """
    Main function to generate random graph data and save to files.
    """
    # Call the graph generation function with the default parameters
    generate_random_graph()

    print("Graph data generated and saved to files.")

# Entry point of the script
if __name__ == '__main__':
    main()
