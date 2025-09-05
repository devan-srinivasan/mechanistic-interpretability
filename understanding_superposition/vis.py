import torch
import numpy as np
import plotly.subplots as sp
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, List, Union
from plotly.subplots import make_subplots

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
    
    for i in range(num_lines):
        fig.add_trace(go.Scatter(
            x=x[i].tolist(), 
            y=y[i].tolist(), 
            mode='lines',
            line=dict(color=colours[i] if colours else None),
            name=f'Line {i}'
        ))
    
    fig.show()

def plot_heatmap(matrices: List[Union[torch.Tensor, np.ndarray]]) -> None:
    """
    Display multiple heatmaps side by side without axes, ensuring square cells.
    
    Args:
        matrices (list): A list of 2D matrices (torch.Tensor or np.ndarray).
    """
    with torch.no_grad():
        # Convert to numpy if torch tensors
        matrices = [m.numpy() if isinstance(m, torch.Tensor) else m for m in matrices]
        
        num_matrices = len(matrices)
        fig = make_subplots(rows=1, cols=num_matrices, horizontal_spacing=0.15)

        for i, mat in enumerate(matrices):
            fig.add_trace(
                go.Heatmap(
                    z=mat, 
                    showscale=False, 
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1
                ),
                row=1, col=i+1
            )

        fig.update_layout(
            showlegend=False,
            width=300 * num_matrices,  # Adjust width dynamically based on number of plots
            height=400,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="white",
        )

        fig.show()