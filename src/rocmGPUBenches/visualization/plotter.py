"""Core plotting functions for benchmark visualization."""

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Any


# GPU color palette (ported from gpu-benches/device_order.py)
GPU_COLOR_PALETTE = [
    "#378ABD",  # Blue
    "#FFB33A",  # Orange
    "#7EC75B",  # Green
    "#DA5252",  # Red
    "#793B67",  # Purple
    "#10CFCC",  # Cyan
    "#FFE100",  # Yellow
    "#09047f",  # Dark blue
    "#296F20",  # Dark green
]

# Default line style
DEFAULT_LINE_STYLE = {
    'linewidth': 2.0,
    'alpha': 1.0,
    'markersize': 5,
    'marker': 'o'
}


def setup_style(style: str = 'bmh'):
    """
    Setup matplotlib style for consistent plot appearance.
    
    Args:
        style: Matplotlib style name (default: 'bmh')
    """
    plt.style.use(style)
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10


def plot_sweep(
    df: pd.DataFrame,
    x: str = 'problem_size',
    y: str = 'primary_metric',
    yerr: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xscale: str = 'linear',
    yscale: str = 'linear',
    group_by: Optional[str] = None,
    figsize: tuple = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    **kwargs
) -> matplotlib.figure.Figure:
    """
    Plot parameter sweep results (line plot with optional error bars).
    
    Args:
        df: DataFrame with benchmark results (from BenchmarkDB.query() or get_sweep_data())
        x: Column name for x-axis (default: 'problem_size')
        y: Column name for y-axis (default: 'primary_metric')
        yerr: Optional column name for error bars (e.g., 'primary_metric_std')
        title: Plot title (auto-generated if None)
        xlabel: X-axis label (uses column name if None)
        ylabel: Y-axis label (uses column name if None)
        xscale: X-axis scale ('linear', 'log', 'log2') (default: 'linear')
        yscale: Y-axis scale ('linear', 'log') (default: 'linear')
        group_by: Optional column to group lines by (e.g., 'gpu_name', 'block_size')
        figsize: Figure size in inches (default: (10, 6))
        save_path: Path to save figure (PNG, SVG, PDF supported)
        show: Whether to display the plot (default: True)
        **kwargs: Additional arguments passed to plt.plot() or plt.errorbar()
    
    Returns:
        matplotlib Figure object
    
    Example:
        >>> db = BenchmarkDB('results.db')
        >>> df = db.get_sweep_data('cache', 'problem_size')
        >>> plot_sweep(df, x='problem_size', y='bandwidth_gbs_mean', 
        ...           yerr='bandwidth_gbs_std', xscale='log2')
    """
    if df.empty:
        raise ValueError("DataFrame is empty, cannot plot")
    
    if x not in df.columns:
        raise ValueError(f"Column '{x}' not found in DataFrame")
    if y not in df.columns:
        raise ValueError(f"Column '{y}' not found in DataFrame")
    
    # Setup style if not already done
    if plt.rcParams['axes.facecolor'] != 'white':
        setup_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Merge default style with user kwargs
    plot_kwargs = {**DEFAULT_LINE_STYLE, **kwargs}
    
    if group_by and group_by in df.columns:
        # Plot multiple lines grouped by a column
        groups = df[group_by].unique()
        for i, group in enumerate(groups):
            group_df = df[df[group_by] == group].sort_values(x)
            color = GPU_COLOR_PALETTE[i % len(GPU_COLOR_PALETTE)]
            
            if yerr and yerr in df.columns:
                ax.errorbar(
                    group_df[x], group_df[y], yerr=group_df[yerr],
                    label=str(group), color=color, **plot_kwargs
                )
            else:
                ax.plot(
                    group_df[x], group_df[y],
                    label=str(group), color=color, **plot_kwargs
                )
    else:
        # Single line plot
        df_sorted = df.sort_values(x)
        if yerr and yerr in df.columns:
            ax.errorbar(
                df_sorted[x], df_sorted[y], yerr=df_sorted[yerr],
                color=GPU_COLOR_PALETTE[0], **plot_kwargs
            )
        else:
            ax.plot(
                df_sorted[x], df_sorted[y],
                color=GPU_COLOR_PALETTE[0], **plot_kwargs
            )
    
    # Set scales
    if xscale == 'log2':
        ax.set_xscale('log', base=2)
    elif xscale in ['log', 'linear']:
        ax.set_xscale(xscale)
    
    if yscale in ['log', 'linear']:
        ax.set_yscale(yscale)
    
    # Labels and title
    ax.set_xlabel(xlabel or x.replace('_', ' ').title())
    ax.set_ylabel(ylabel or y.replace('_', ' ').title())
    
    if title:
        ax.set_title(title)
    elif 'benchmark_type' in df.columns:
        benchmark = df['benchmark_type'].iloc[0]
        ax.set_title(f'{benchmark.title()} Benchmark - Parameter Sweep')
    
    # Legend if grouped
    if group_by:
        ax.legend()
    
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_comparison(
    df: pd.DataFrame,
    metric: str = 'primary_metric',
    group_by: str = 'gpu_name',
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    **kwargs
) -> matplotlib.figure.Figure:
    """
    Plot bar chart comparing metrics across different configurations or GPUs.
    
    Args:
        df: DataFrame with benchmark results
        metric: Column name for the metric to compare (default: 'primary_metric')
        group_by: Column to group bars by (default: 'gpu_name')
        title: Plot title (auto-generated if None)
        ylabel: Y-axis label (uses metric name if None)
        figsize: Figure size in inches (default: (10, 6))
        save_path: Path to save figure
        show: Whether to display the plot (default: True)
        **kwargs: Additional arguments passed to plt.bar()
    
    Returns:
        matplotlib Figure object
    
    Example:
        >>> df = db.query(benchmark='cache', problem_size=1024)
        >>> plot_comparison(df, metric='primary_metric', group_by='gpu_name')
    """
    if df.empty:
        raise ValueError("DataFrame is empty, cannot plot")
    
    if metric not in df.columns:
        raise ValueError(f"Column '{metric}' not found in DataFrame")
    if group_by not in df.columns:
        raise ValueError(f"Column '{group_by}' not found in DataFrame")
    
    # Setup style
    if plt.rcParams['axes.facecolor'] != 'white':
        setup_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Aggregate data by group
    grouped = df.groupby(group_by)[metric].mean().sort_values(ascending=False)
    
    # Create bar chart
    x_pos = np.arange(len(grouped))
    colors = [GPU_COLOR_PALETTE[i % len(GPU_COLOR_PALETTE)] for i in range(len(grouped))]
    
    bars = ax.bar(x_pos, grouped.values, color=colors, **kwargs)
    
    # Labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(grouped.index, rotation=45, ha='right')
    ax.set_ylabel(ylabel or metric.replace('_', ' ').title())
    
    if title:
        ax.set_title(title)
    elif 'benchmark_type' in df.columns:
        benchmark = df['benchmark_type'].iloc[0]
        ax.set_title(f'{benchmark.title()} Benchmark - Comparison')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_heatmap(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str = 'primary_metric',
    title: Optional[str] = None,
    figsize: tuple = (10, 8),
    cmap: str = 'viridis',
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    **kwargs
) -> matplotlib.figure.Figure:
    """
    Plot 2D heatmap for dual-parameter sweeps.
    
    Args:
        df: DataFrame with benchmark results
        x: Column name for x-axis parameter
        y: Column name for y-axis parameter
        z: Column name for metric to visualize (default: 'primary_metric')
        title: Plot title (auto-generated if None)
        figsize: Figure size in inches (default: (10, 8))
        cmap: Colormap name (default: 'viridis')
        save_path: Path to save figure
        show: Whether to display the plot (default: True)
        **kwargs: Additional arguments passed to plt.imshow()
    
    Returns:
        matplotlib Figure object
    
    Example:
        >>> # For a sweep across problem_size and block_size
        >>> df = db.query(benchmark='cache')
        >>> plot_heatmap(df, x='problem_size', y='block_size', z='primary_metric')
    """
    if df.empty:
        raise ValueError("DataFrame is empty, cannot plot")
    
    for col in [x, y, z]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Setup style
    if plt.rcParams['axes.facecolor'] != 'white':
        setup_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Pivot data for heatmap
    pivot_df = df.pivot_table(values=z, index=y, columns=x, aggfunc='mean')
    
    # Create heatmap
    im = ax.imshow(pivot_df.values, cmap=cmap, aspect='auto', **kwargs)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_xticklabels(pivot_df.columns)
    ax.set_yticklabels(pivot_df.index)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(z.replace('_', ' ').title(), rotation=270, labelpad=20)
    
    # Labels and title
    ax.set_xlabel(x.replace('_', ' ').title())
    ax.set_ylabel(y.replace('_', ' ').title())
    
    if title:
        ax.set_title(title)
    elif 'benchmark_type' in df.columns:
        benchmark = df['benchmark_type'].iloc[0]
        ax.set_title(f'{benchmark.title()} Benchmark - 2D Parameter Sweep')
    
    fig.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def format_data_size_axis(ax, unit='auto', base_unit_bytes=4):
    """
    Format axis to show data sizes in KB/MB with smart labels.
    
    Args:
        ax: Matplotlib axis object
        unit: 'KB', 'MB', or 'auto' for automatic selection
        base_unit_bytes: Bytes per element (default: 4 for float32)
    
    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot(problem_sizes, bandwidths)
        >>> format_data_size_axis(ax)
    """
    import matplotlib.ticker as ticker
    
    def size_formatter(x, pos):
        """Convert element count to KB/MB"""
        kb = (x * base_unit_bytes) / 1024
        if kb < 1024:
            return f"{kb:g} KB"
        else:
            return f"{kb/1024:g} MB"
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(size_formatter))
    return ax


def plot_gpu_comparison_sweep(
    df: pd.DataFrame,
    x: str = 'problem_size',
    y: str = 'primary_metric',
    title: Optional[str] = None,
    element_size_bytes: int = 4,
    figsize: tuple = (12, 7),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    **kwargs
) -> matplotlib.figure.Figure:
    """
    Plot parameter sweep comparison across multiple GPUs.
    Convenience wrapper around plot_sweep with gpu_name grouping.
    
    Args:
        df: DataFrame with results from multiple GPUs
        x: Parameter column (default: 'problem_size')
        y: Metric column (default: 'primary_metric')  
        title: Plot title
        element_size_bytes: Bytes per element for size formatting (default: 4 for float32)
        figsize: Figure size (default: (12, 7))
        save_path: Path to save figure
        show: Whether to display plot
        **kwargs: Additional arguments for plot_sweep()
    
    Returns:
        matplotlib Figure object
    
    Example:
        >>> # Results from MI325X, MI300X, H100
        >>> df = db.query(benchmark='cache')
        >>> plot_gpu_comparison_sweep(df, x='problem_size', y='primary_metric')
    """
    if 'gpu_name' not in df.columns:
        raise ValueError("DataFrame must have 'gpu_name' column for GPU comparison")
    
    # Use plot_sweep with gpu_name grouping
    fig = plot_sweep(
        df,
        x=x,
        y=y,
        group_by='gpu_name',
        title=title or 'GPU Performance Comparison',
        xlabel=f'Data Volume per CU/SM (elements)',
        figsize=figsize,
        save_path=save_path,
        show=show,
        **kwargs
    )
    
    # Format x-axis to show KB/MB
    ax = fig.axes[0]
    format_data_size_axis(ax, base_unit_bytes=element_size_bytes)
    ax.set_xlabel('Data Volume per CU/SM')
    
    if save_path and not show:
        # Re-save with updated formatting
        fig.savefig(save_path, bbox_inches='tight')
    
    return fig
