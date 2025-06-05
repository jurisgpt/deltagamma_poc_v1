"""
Smoke test for the EveryCure core pipeline using a synthetic graph.
This test generates a random graph, extracts embeddings via the GAT autoencoder,
and trains an XGBoost classifier to ensure end-to-end connectivity.
"""

import pytest

# Skip entire test if essential libraries are not installed
pytest.importorskip(
    "torch_geometric",
    reason="torch_geometric not installed; skipping smoke pipeline test",
)
pytest.importorskip(
    "xgboost", reason="xgboost not installed; skipping smoke pipeline test"
)

import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def test_smoke_pipeline(tmp_path, monkeypatch):
    # Switch to a temporary directory for test outputs
    monkeypatch.chdir(tmp_path)
    # Allow multiple OpenMP runtimes for subprocesses (macOS workaround)
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")

    # Locate the scripts directory (project root)
    repo_root = Path(__file__).resolve().parent.parent
    scripts_dir = repo_root

    # 1) Generate a random graph
    subprocess.run(
        [sys.executable, str(scripts_dir / "build_test_graph.py")],
        check=True,
    )

    # Ensure the graph files were created
    graph_file = tmp_path / "Output_Graph"
    node_ids_file = tmp_path / "node_ids.csv"
    class_labels = tmp_path / "Class_Labels.csv"
    assert graph_file.exists()
    assert node_ids_file.exists()
    assert class_labels.exists()

    # 2) Extract embeddings (1 epoch for speed)
    embeddings_file = tmp_path / "test_embeddings.csv"
    subprocess.run(
        [
            sys.executable,
            str(scripts_dir / "embedding_extractor.py"),
            "--graph_file",
            str(graph_file),
            "--node_ids_file",
            str(node_ids_file),
            "--embedding_output",
            str(embeddings_file),
            "--hidden_channels",
            "4",
            "--epochs",
            "1",
            "--lr",
            "0.01",
        ],
        check=True,
    )
    assert embeddings_file.exists()
    emb_df = pd.read_csv(embeddings_file)
    assert set(emb_df.columns) == {"id", "topological_embedding"}

    # 3) Train the XGBoost classifier
    model_file = tmp_path / "test_model.pkl"
    subprocess.run(
        [
            sys.executable,
            str(scripts_dir / "train_model.py"),
            "--embedding_file",
            str(embeddings_file),
            "--label_file",
            str(class_labels),
            "--model_file",
            str(model_file),
        ],
        check=True,
    )
    assert model_file.exists()

    # Load the model and do a dummy prediction
    model = joblib.load(model_file)
    labels_df = pd.read_csv(class_labels)
    emb_df = pd.read_csv(embeddings_file)
    src_id, tgt_id = labels_df.iloc[0][["source", "target"]]
    src_emb = emb_df.loc[emb_df["id"] == src_id, "topological_embedding"].iloc[0]
    tgt_emb = emb_df.loc[emb_df["id"] == tgt_id, "topological_embedding"].iloc[0]
    src_vec = np.array([float(x) for x in src_emb.split()])
    tgt_vec = np.array([float(x) for x in tgt_emb.split()])
    sample = np.concatenate([src_vec, tgt_vec])[None, :]
    pred = model.predict(sample)
    assert pred.shape == (1,)
