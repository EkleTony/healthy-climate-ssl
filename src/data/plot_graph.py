import argparse
import os
import torch
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot kNN graph over region coordinates.")
    parser.add_argument("--dataset_dir", type=str, default="data/v1_seed7",
                        help="Dataset directory containing regions.pt and graph.pt")
    parser.add_argument("--stride", type=int, default=5,
                        help="Plot every N-th edge to reduce clutter (default=5)")
    parser.add_argument("--no_show", action="store_true",
                        help="Do not display the plot window")
    args = parser.parse_args()

    regions_path = os.path.join(args.dataset_dir, "regions.pt")
    graph_path = os.path.join(args.dataset_dir, "graph.pt")

    if not os.path.exists(regions_path):
        raise FileNotFoundError(f"Missing: {regions_path}")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Missing: {graph_path}")

    regions = torch.load(regions_path)
    graph = torch.load(graph_path)

    coords = regions["coords"].cpu().numpy()
    state_id = regions["state_id"].cpu().numpy()
    edge_index = graph["edge_index"].cpu().numpy()
    k = graph.get("k", None)

    plt.figure(figsize=(8, 8))

    stride = max(1, int(args.stride))
    for i in range(0, edge_index.shape[1], stride):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        plt.plot([coords[src, 0], coords[dst, 0]],
                 [coords[src, 1], coords[dst, 1]],
                 color="lightgray", linewidth=0.2)

    plt.scatter(coords[:, 0], coords[:, 1], c=state_id, cmap="tab10", s=10)

    title = "kNN Spatial Graph"
    if k is not None:
        title += f" (k={k})"
    title += f" | edges={edge_index.shape[1]}"
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")

    # Save in same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "graph_plot.png")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved plot to: {save_path}")

    # if not args.no_show:
    #     plt.show()


if __name__ == "__main__":
    main()
