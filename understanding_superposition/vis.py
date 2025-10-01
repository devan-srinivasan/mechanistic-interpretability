import torch
import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Union
import numpy as np

def plot_lines(x: torch.Tensor, y: torch.Tensor, colours: Optional[List[str]] = None) -> None:
    """
    Plot multiple lines using Plotly from PyTorch tensors.
    
    Args:
        x (torch.Tensor): A 2D tensor where each row represents x-coordinates of a line.
        y (torch.Tensor): A 2D tensor where each row represents y-coordinates of a line.
        colours (list, optional): A list of colors for each line.
    """
    fig = go.Figure()
    num_lines = x.shape[0]
    assert(len(colours) == num_lines if colours else True), "Length of colours must match number of lines"
    
    for i in range(num_lines):
        fig.add_trace(go.Scatter(
            x=x[i].tolist(), 
            y=y[i].tolist(), 
            mode='lines',
            line=dict(color=colours[i] if colours else None),
            name=f'Line {i}'
        ))

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title="X",
        yaxis_title="Y"
    )
    fig.show()


def plot_matrices(matrices: List[Union[torch.Tensor, np.ndarray]]) -> None:
    """
    Plot multiple matrices as tiled heatmaps using Plotly.
    Automatically arranges them in a grid with a red-blue color scale [-1, 1].
    
    Args:
        matrices (list): List of 2D matrices (torch.Tensor or np.ndarray)
    """
    if len(matrices) == 0:
        return
    
    # Convert to numpy
    matrices = [m.numpy() if isinstance(m, torch.Tensor) else m for m in matrices]

    num_matrices = len(matrices)
    cols = math.ceil(math.sqrt(num_matrices))
    rows = math.ceil(num_matrices / cols)

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"Matrix {i}" for i in range(num_matrices)])

    for i, matrix in enumerate(matrices):
        row = i // cols + 1
        col = i % cols + 1
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                showscale=(i == num_matrices - 1)  # only show scale for last plot to save space
            ),
            row=row,
            col=col
        )

    fig.update_layout(
        template="plotly_white",
        height=max(300, rows * 300),
        width=max(300, cols * 300),
        margin=dict(l=10, r=10, t=40, b=10),
    )

    # Remove tick labels for a cleaner look
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    fig.show()


def plot_matrix(matrix: Union[torch.Tensor, np.ndarray]) -> None:
    """
    Plot a single matrix as a heatmap using Plotly.
    Automatically scales the figure size to fit the matrix.
    """
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.numpy()

    n_rows, n_cols = matrix.shape
    cell_size = 40
    width = max(300, min(1000, cell_size * n_cols))
    height = max(300, min(1000, cell_size * n_rows))

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            showscale=True
        )
    )

    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="white",
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )
    fig.show()


def plot_clusters(labels: List[str], cluster_heads: List[int]) -> None:
    """
    Plot clusters as tight pools at geometrically spaced points on a circle.
    Each cluster's points are tightly grouped at a unique location on the circle.
    Hovering shows the label.
    
    Args:
        labels (List[str]): List of labels for each point.
        cluster_heads (List[int]): List of cluster assignments (same length as labels).
    """
    import plotly.express as px

    assert(len(labels) == len(cluster_heads)), "Labels and cluster_heads must be the same length"

    # Find unique clusters
    unique_clusters = sorted(set(cluster_heads))
    n_clusters = len(unique_clusters)

    # Arrange cluster centers on a circle
    cluster_centers = [
        (np.cos(2 * np.pi * i / n_clusters), np.sin(2 * np.pi * i / n_clusters))
        for i in range(n_clusters)
    ]

    # Assign points for each cluster: tightly around its center
    x, y, text, color = [], [], [], []
    cluster_spread = 0.05  # controls tightness of cluster
    rng = np.random.default_rng(42)
    for idx, cluster in enumerate(unique_clusters):
        center_x, center_y = cluster_centers[idx]
        indices = [i for i, c in enumerate(cluster_heads) if c == cluster]
        for i in indices:
            # Add small random jitter so points don't overlap exactly
            px_ = center_x + cluster_spread * rng.normal()
            py_ = center_y + cluster_spread * rng.normal()
            x.append(px_)
            y.append(py_)
            text.append(labels[i])
            color.append(str(cluster))

    fig = px.scatter(
        x=x, y=y, color=color, hover_name=text,
        labels={'color': 'Cluster'},
        title="Clusters Geometrically Spaced on a Circle"
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(template="plotly_white")
    fig.show()

# Generate dummy data: 10 clusters, 10 fruits, each fruit replicated 2800 times
# fruits = ["apple", "banana", "cherry", "date", "fig", "grape", "kiwi", "lemon", "mango", "nectarine"]
# num_clusters = 10
# replications = 2200

# labels = fruits * replications  # 10 fruits * 2200 = 22,000 labels
# cluster_heads = [i for i in range(num_clusters) for _ in range(replications)]  # 10 clusters, each with 2200 points

# plot_clusters(labels, cluster_heads)