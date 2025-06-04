#!/usr/bin/env python3
import torch
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import pandas as pd
import argparse


class GATModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_types_count, heads=8):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels, heads=heads, concat=True)
        self.conv2 = GATConv(out_channels * heads, out_channels, heads=1, concat=False)
        
        # Embedding layer for edge attributes (edge types)
        self.edge_embedding = torch.nn.Embedding(edge_types_count, out_channels)  # Map edge types to embeddings

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Look up the edge embeddings for the edge types
        edge_embeddings = self.edge_embedding(edge_attr)  # Get embeddings for edge attributes
        #print(edge_embeddings)  # Print the learned edge embeddings for sense check
        edge_attr = edge_embeddings  # Use these embeddings as the new edge features
        


        # Perform the GAT convolutions
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)

        return x  # Return only the node embeddings, not edge embeddings




def extract_embeddings(graph_object,node_ids, embedding_output):
    """
    Extract embeddings from a graph using GAT and save them to a file with node IDs.
    """
    # Load the graph object 
    data = torch.load(graph_object)

    # Extract node features from the loaded graph
    node_features = data.x

    # Extract edge attributes (edge types)
    edge_types = data.edge_attr

    # Extract unique edge types (if not already known)
    unique_edge_types = sorted(set(edge_types.numpy()))  # Convert tensor to numpy for set operation


    # Instantiate the model
    model = GATModel(in_channels=node_features.shape[1], out_channels=4, edge_types_count=len(unique_edge_types))

    # Set the model to evaluation mode
    model.eval()

    # Run the model to get node embeddings
    with torch.no_grad():  # Disable gradient tracking during inference
        node_embeddings = model(data)

    # Debug: Print the shape of the node embeddings
    print(f"Node embeddings shape: {node_embeddings.shape}")

    # Convert the node embeddings to a numpy array for saving
    node_embeddings = node_embeddings.cpu().numpy()

    # Create a DataFrame for saving the embeddings
    embeddings_df = pd.DataFrame(node_embeddings)


    # Load the node IDs from the CSV
    node_ids_df = pd.read_csv(node_ids)
    node_ids = node_ids_df['id']


    # Add node IDs to the embeddings DataFrame
    embeddings_df['id'] = node_ids


    # Create the 'topological_embedding' column by joining the embeddings into a single string per node
    # Use only the node embeddings, exclude edge attributes
    embeddings_df['topological_embedding'] = embeddings_df.iloc[:, :-1].apply(lambda row: ' '.join(row.astype(str)), axis=1)

    # Keep only the 'id' and 'topological_embedding' columns
    embeddings_df = embeddings_df[['id', 'topological_embedding']]

    # Save the embeddings to a CSV
    embeddings_df.to_csv(embedding_output, index=False)
    print(f"Embeddings saved to {embedding_output}")


def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Extract node embeddings using a GAT model.")
    parser.add_argument('--graph_object', type=str, required=True, help="Path to the pyTorch graph object")
    parser.add_argument('--node_ids', type=str, required=True, help="Path to the node IDs for graph object CSV file")
    parser.add_argument('--embedding_output', type=str, required=True, help="Path to save the node embeddings CSV file")
    
    args = parser.parse_args()

    # Extract embeddings and save to file
    extract_embeddings(args.graph_object, args.node_ids, args.embedding_output)

if __name__ == "__main__":
    main()
