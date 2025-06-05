#!/usr/bin/env python3
"""
Orchestrate the full EveryCure pipeline: graph construction, embedding extraction, and classification.
"""
import argparse

from graph_builder import build_knowledge_graph
from embedding_extractor import extract_embeddings
from train_model import train_ml_model


def run_full_pipeline(
    node_file: str,
    edge_file: str,
    label_file: str,
    graph_file: str,
    node_ids_file: str,
    embedding_file: str,
    model_file: str,
    hidden_channels: int,
    epochs: int,
    lr: float,
) -> None:
    """
    Execute all steps: build graph, train GAT embeddings, and train classifier.
    """
    print("[1/3] Building knowledge graph...")
    build_knowledge_graph(node_file, edge_file, graph_file, node_ids_file)

    print("[2/3] Extracting node embeddings via GAT autoencoder...")
    extract_embeddings(
        graph_file,
        node_ids_file,
        embedding_file,
        hidden_channels=hidden_channels,
        epochs=epochs,
        lr=lr,
    )

    print("[3/3] Training XGBoost classifier on embeddings...")
    train_ml_model(embedding_file, label_file, model_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EveryCure full pipeline or its individual stages"
    )
    parser.add_argument(
        "--task",
        choices=["full_pipeline", "build_graph", "extract_embeddings", "train_model"],
        required=True,
        help="Pipeline stage to run",
    )

    # Paths for data and outputs
    parser.add_argument("--node_file", help="Path to input node CSV")
    parser.add_argument("--edge_file", help="Path to input edge CSV")
    parser.add_argument("--graph_file", help="Output path for graph object (.pt)")
    parser.add_argument(
        "--node_ids_file", default="node_ids.csv", help="Path for node ID mapping CSV"
    )
    parser.add_argument("--embedding_file", help="Output path for embeddings CSV")
    parser.add_argument("--label_file", help="Path to label CSV for classifier")
    parser.add_argument("--model_file", help="Output path for trained model (.pkl)")

    # Hyperparameters for embedding extraction and training
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=16,
        help="Hidden dimension for GAT layers",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs for GAT training"
    )
    parser.add_argument(
        "--lr", type=float, default=0.005, help="Learning rate for GAT training"
    )

    args = parser.parse_args()

    if args.task == "full_pipeline":
        run_full_pipeline(
            args.node_file,
            args.edge_file,
            args.label_file,
            args.graph_file,
            args.node_ids_file,
            args.embedding_file,
            args.model_file,
            args.hidden_channels,
            args.epochs,
            args.lr,
        )
    elif args.task == "build_graph":
        build_knowledge_graph(
            args.node_file, args.edge_file, args.graph_file, args.node_ids_file
        )
    elif args.task == "extract_embeddings":
        extract_embeddings(
            args.graph_file,
            args.node_ids_file,
            args.embedding_file,
            hidden_channels=args.hidden_channels,
            epochs=args.epochs,
            lr=args.lr,
        )
    elif args.task == "train_model":
        train_ml_model(args.embedding_file, args.label_file, args.model_file)


if __name__ == "__main__":
    main()
