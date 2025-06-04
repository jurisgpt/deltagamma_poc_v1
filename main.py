import argparse
import os
from graph_builder import build_knowledge_graph
from embedding_extractor import extract_embeddings
from ml_model import train_ml_model

def run_full_pipeline(node_file, edge_file, label_file, graph_output, embedding_output):
    """
    Run the full pipeline from graph construction, embedding extraction to model training.
    """
    print("Building knowledge graph...")
    build_knowledge_graph(node_file, edge_file, graph_output)

    print("Extracting node embeddings...")
    extract_embeddings(graph_output, embedding_output)

    print("Training the classification model...")
    train_ml_model(embedding_output, label_file)

def main(args):
    if args.task == 'full_pipeline':
        run_full_pipeline(args.node_file, args.edge_file, args.label_file, args.graph_file, args.embedding_file)
    elif args.task == 'build_graph':
        build_knowledge_graph(args.node_file, args.edge_file, args.graph_file)
    elif args.task == 'extract_embeddings':
        extract_embeddings(args.graph_file, args.embedding_file)
    elif args.task == 'train_model':
        train_ml_model(args.embedding_file, args.label_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Knowledge graph and pair prediction.")
    
    # General task argument
    parser.add_argument('--task', choices=['full_pipeline', 'build_graph', 'extract_embeddings', 'train_model'], required=True, help='Task to run')
    
    # Task-specific arguments
    parser.add_argument('--node_file', help='Path to the node CSV file')
    parser.add_argument('--edge_file', help='Path to the edge CSV file')
    parser.add_argument('--graph_file', help='Path to the saved graph file (.pt)')
    parser.add_argument('--embedding_file', help='Path to the saved embeddings file (.csv)')
    parser.add_argument('--label_file', help='Path to the label CSV file')

    args = parser.parse_args()
    main(args)
