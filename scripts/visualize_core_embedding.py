#!/usr/bin/env python
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def main():
    parser = argparse.ArgumentParser(description="Visualize core embeddings with t-SNE.")
    parser.add_argument("--embedding-file", type=str, default=None, help="Path to npy file with z_core (B,D).")
    parser.add_argument("--labels-file", type=str, default=None, help="Path to npy file with labels (B,).")
    parser.add_argument("--save-path", type=str, default="logs/core_tsne.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    if args.embedding_file and os.path.exists(args.embedding_file):
        z = np.load(args.embedding_file)
    else:
        z = np.random.randn(200, 32)
    if args.labels_file and os.path.exists(args.labels_file):
        labels = np.load(args.labels_file)
    else:
        labels = np.random.randint(0, 10, size=(z.shape[0],))

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=500, init="random", random_state=42)
    coords = tsne.fit_transform(z)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab20", s=10, alpha=0.8)
    plt.title("Core Embedding t-SNE")
    plt.colorbar(scatter, shrink=0.7)
    plt.tight_layout()
    plt.savefig(args.save_path)
    plt.close()
    print(f"[LOG] Saved core embedding t-SNE to {args.save_path}")


if __name__ == "__main__":
    main()
