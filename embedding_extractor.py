#!/usr/bin/env python3
"""
Train a graph autoencoder (using GAT) to generate node embeddings and save them to CSV.
"""
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import GAE, GATConv


class GATEncoder(torch.nn.Module):
    """
    Graph Attention Network (GAT) encoder for unsupervised link reconstruction.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        # First GAT layer
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        # Second GAT layer to obtain final embeddings
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        # Apply first attention layer and activation
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        # Apply second attention layer
        return self.conv2(x, edge_index)


def extract_embeddings(graph_file: str,
                       node_ids_file: str,
                       embedding_output: str,
                       hidden_channels: int = 16,
                       epochs: int = 100,
                       lr: float = 0.005):
    """
    Train a GAT-based autoencoder to reconstruct edges, then save node embeddings.

    Parameters:
    - graph_file: path to the PyG Data object (.pt) containing x, edge_index, edge_attr.
    - node_ids_file: CSV with column 'id' mapping node index to original ID.
    - embedding_output: path to save embeddings CSV (id, topological_embedding).
    - hidden_channels: hidden dimension for the GAT layer.
    - epochs: number of training epochs.
    - lr: learning rate for optimizer.
    """
    # Load graph data
    data: Data = torch.load(graph_file)
    x, edge_index = data.x, data.edge_index

    # Initialize GAT encoder and autoencoder wrapper
    encoder = GATEncoder(in_channels=x.size(1),
                         hidden_channels=hidden_channels,
                         out_channels=hidden_channels)
    model = GAE(encoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop: reconstruct edges by minimizing reconstruction loss
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        z = model.encode(x, edge_index)
        loss = model.recon_loss(z, edge_index)
        loss.backward()
        optimizer.step()
        if epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch:03d}/{epochs}, Reconstruction Loss: {loss:.4f}")

    # Generate final embeddings in evaluation mode
    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index)

    # Convert embeddings to pandas DataFrame
    emb_np = z.cpu().numpy()
    df = pd.DataFrame(emb_np)

    # Load original node IDs to match embedding rows
    ids_df = pd.read_csv(node_ids_file)
    df['id'] = ids_df['id']

    # Combine numeric embeddings into space-separated string for each node
    df['topological_embedding'] = df.drop(columns='id') \
        .astype(str).agg(' '.join, axis=1)
    out_df = df[['id', 'topological_embedding']]
    out_df.to_csv(embedding_output, index=False)
    print(f"Embeddings saved to {embedding_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract node embeddings via GAT autoencoder"
    )
    parser.add_argument('--graph_file', required=True,
                        help='Path to graph object file (.pt)')
    parser.add_argument('--node_ids_file', required=True,
                        help='CSV file mapping node indices to original IDs')
    parser.add_argument('--embedding_output', required=True,
                        help='Output CSV path for embeddings')
    parser.add_argument('--hidden_channels', type=int, default=16,
                        help='Hidden dimension size for GAT layers')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate for training')
    args = parser.parse_args()

    extract_embeddings(
        args.graph_file,
        args.node_ids_file,
        args.embedding_output,
        hidden_channels=args.hidden_channels,
        epochs=args.epochs,
        lr=args.lr
    )


if __name__ == '__main__':
    main()