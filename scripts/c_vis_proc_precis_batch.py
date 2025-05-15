import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from typing import Optional, List, Dict
from matplotlib import cm

# ---------------------------
# Factory functions for global colour maps
# ---------------------------
def make_model_energy_colours(models: List[str], cmap_name: str = 'tab10') -> Dict[str, tuple]:
    """
    Create a consistent mapping from model names to colours for energy plots.
    """
    cmap = cm.get_cmap(cmap_name, len(models))
    return {m: cmap(i) for i, m in enumerate(models)}


def make_model_throughput_colours(models: List[str], cmap_name: str = 'Reds') -> Dict[str, tuple]:
    """
    Create a consistent mapping from model names to colours for throughput plots.
    """
    cmap = cm.get_cmap(cmap_name, len(models))
    return {m: cmap(i) for i, m in enumerate(models)}

# ---------------------------
# Internal plotting helper
# ---------------------------
def _plot_with_band(
    ax,
    x,
    mean_vals,
    std_vals,
    color,
    linestyle,
    marker,
    label,
    alpha_line,
    alpha_band,
    alpha_scatter,
    plot_band,
    plot_raw
):
    if plot_raw:
        ax.scatter(x, mean_vals,
                   marker=marker,
                   alpha=alpha_scatter,
                   color=color)
    ax.plot(x, mean_vals,
            linestyle=linestyle,
            marker=marker,
            color=color,
            label=label,
            alpha=alpha_line)
    if plot_band:
        ax.fill_between(x,
                        mean_vals - std_vals,
                        mean_vals + std_vals,
                        color=color,
                        alpha=alpha_band)

# ---------------------------
# Generic param vs metric
# ---------------------------
def plot_param_vs_metric(
    df: pd.DataFrame,
    param_col: str,
    ax1: str = 'energy_per_token',
    ax2: Optional[str] = None,
    normalise_axes: Optional[List[str]] = None,
    plot_mean: bool = True,
    plot_band: bool = True,
    plot_raw: bool = True,
    add_baseline_energy: bool = False,
    add_baseline_throughput: bool = False,
    models: Optional[List[str]] = None,
    energy_colours: Optional[Dict[str, tuple]] = None,
    throughput_colours: Optional[Dict[str, tuple]] = None
):
    # metric lookup
    metric_map = {
        'energy_per_token':         {'col': 'energy_per_token_kwh',
                                     'label': 'Energy per Token (kWh)',
                                     'legend': 'Energy'},
        'throughput_tokens_per_sec':{'col': 'throughput_tokens_per_sec',
                                     'label': 'Throughput (tokens/sec)',
                                     'legend': 'Throughput'},
        'gpu_utilization_proc_all':{ 'col': 'gpu_utilization_proc_all',
                                     'label': 'GPU Utilisation (mean over processes))',
                                     'legend': 'GPU Utilisation'}
    }
    if ax1 not in metric_map:
        raise ValueError(f"ax1 must be one of {list(metric_map.keys())}")
    if ax2 is not None and ax2 not in metric_map:
        raise ValueError(f"ax2 must be one of {list(metric_map.keys())} or None")

    normalise_axes = normalise_axes or []
    norm1 = 'ax1' in normalise_axes
    norm2 = 'ax2' in normalise_axes

    # determine which models to loop
    if models is None and 'model' in df.columns:
        models = df['model'].dropna().unique().tolist()
    elif models is None:
        models = [None]

    # default colour maps if none provided
    if energy_colours is None:
        energy_colours = make_model_energy_colours(models)
    if throughput_colours is None:
        throughput_colours = make_model_throughput_colours(models)

    linestyles    = ['-', '--', '-.', (0,(5,1))]
    markers       = ['o','D','^','s']
    alpha_line    = 1.0
    alpha_scatter = 0.2
    alpha_band    = 0.2

    # human-readable axis labels
    xlabel_map = {
        'batch_size': 'Batch Size',
        'num_processes': 'Number of Distributed Processes',
        'precision': 'Numerical Precision'
    }
    xlabel = xlabel_map.get(param_col, param_col.replace('_', ' ').title())

    # prepare display metrics (strip units, hyphenate)
    raw_ax1_label = metric_map[ax1]['label'].split('(')[0].strip()
    display_ax1  = raw_ax1_label.replace(' ', '-').lower().capitalize()
    if ax2:
        raw_ax2_label = metric_map[ax2]['label'].split('(')[0].strip()
        display_ax2  = raw_ax2_label.replace(' ', '-').lower().capitalize()

    fig, ax_left = plt.subplots(figsize=(8, 6))
    ax_right = ax_left.twinx() if ax2 else None

    # set axis labels
    ax_left.set_xlabel(xlabel)
    ax_left.set_ylabel(metric_map[ax1]['label'], color=energy_colours[list(energy_colours.keys())[0]])
    ax_left.tick_params(axis='y', colors=energy_colours[list(energy_colours.keys())[0]])
    ax_left.xaxis.set_major_locator(MaxNLocator(integer=True))
    if ax_right:
        ax_right.set_ylabel(metric_map[ax2]['label'], color=throughput_colours[list(throughput_colours.keys())[0]])
        ax_right.tick_params(axis='y', colors=throughput_colours[list(throughput_colours.keys())[0]])

    last_stats1 = None

    for i, model in enumerate(models):
        sub = df[df['model'] == model] if model is not None else df

        stats1 = sub.groupby(param_col)[metric_map[ax1]['col']].agg(['mean', 'std'])
        last_stats1 = stats1
        xs = (stats1.index.astype(float)
              if is_numeric_dtype(stats1.index)
              else np.arange(len(stats1)))
        mean1 = stats1['mean'].values
        std1  = stats1['std'].fillna(0).values

        if ax2:
            stats2 = sub.groupby(param_col)[metric_map[ax2]['col']].agg(['mean', 'std'])
            mean2  = stats2['mean'].values
            std2   = stats2['std'].fillna(0).values

        if norm1 and len(mean1):
            base1 = mean1[0]
            mean1 /= base1; std1 /= base1
            ax_left.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _: f"{v:.2f}x"))
        if ax2 and norm2 and len(mean2):
            base2 = mean2[0]
            mean2 /= base2; std2 /= base2
            ax_right.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _: f"{v:.2f}x"))

        ls  = linestyles[i % len(linestyles)]
        mk  = markers[i % len(markers)]
        lbl = str(model) if model is not None else 'all'

        # plot energy curve
        _plot_with_band(
            ax_left, xs, mean1, std1,
            color=energy_colours.get(model, 'gray'),
            linestyle=ls, marker=mk,
            label=f"{lbl} {metric_map[ax1]['legend']}",
            alpha_line=alpha_line,
            alpha_scatter=alpha_scatter,
            alpha_band=alpha_band,
            plot_band=plot_band,
            plot_raw=plot_raw
        )
        if add_baseline_energy and len(mean1):
            base = 1.0 if norm1 else mean1[0]
            ax_left.axhline(base, linestyle=':',
                            color=energy_colours.get(model, 'gray'),
                            alpha=0.4)
            ax_left.text(xs[-1], base,
                         f"Energy baseline ({xlabel}: {stats1.index[0]})",
                         ha='right', va='bottom', fontsize='small', alpha=0.4)

        # plot throughput curve
        if ax2:
            _plot_with_band(
                ax_right, xs, mean2, std2,
                color=throughput_colours.get(model, 'gray'),
                linestyle=ls, marker=mk,
                label=f"{lbl} {metric_map[ax2]['legend']}",
                alpha_line=alpha_line,
                alpha_scatter=alpha_scatter,
                alpha_band=alpha_band,
                plot_band=plot_band,
                plot_raw=plot_raw
            )
            if add_baseline_throughput and len(mean2):
                base2 = 1.0 if norm2 else mean2[0]
                ax_right.axhline(base2, linestyle=':',
                                 color=throughput_colours.get(model, 'gray'), alpha=0.4)
                ax_right.text(xs[-1], base2,
                              f"Throughput baseline ({xlabel}: {stats2.index[0]})",
                              ha='right', va='bottom', fontsize='small', alpha=0.4)

    # set categorical ticks if needed
    if last_stats1 is not None and not is_numeric_dtype(last_stats1.index):
        ax_left.set_xticks(np.arange(len(last_stats1)))
        ax_left.set_xticklabels(last_stats1.index.tolist())

    # legend & styling
    handles, labels = ax_left.get_legend_handles_labels()
    if ax_right:
        h2, l2 = ax_right.get_legend_handles_labels()
        handles += h2; labels += l2
    ax_left.legend(handles, labels, loc='best')
    if ax_right is None:
        ax_left.spines['right'].set_visible(False)
    ax_left.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax_left.grid(True, axis='x', linestyle=':', alpha=0.2)

    # build and set title (hyphenated, no units)
    if ax2:
        title = f"{display_ax1} & {display_ax2} vs {xlabel.replace(' ', '-')}"
    else:
        title = f"{display_ax1} vs {xlabel.replace(' ', '-')}"
    suffix = "\n(Normalised)" if norm1 or norm2 else "\n(Absolute Values)"
    plt.title(title + suffix)
    plt.tight_layout()
    plt.show()

# ---------------------------
# Wrappers 
# ---------------------------
def plot_batching(
    dfs,
    MODEL_COLOURS: Optional[Dict[str, tuple]] = None,
    **kwargs
):
    if isinstance(dfs, dict):
        df = dfs['batching'].copy().rename(
            columns={'batch_size___fixed_batching': 'batch_size'}
        )
    elif isinstance(dfs, pd.DataFrame):
        df = dfs.copy().rename(
            columns={'batch_size___fixed_batching': 'batch_size'}
        )
    else:
        raise ValueError("dfs must be a dict or a DataFrame")
    
    energy_colours     = {m: c['energy']     for m, c in MODEL_COLOURS.items()}
    throughput_colours = {m: c['throughput'] for m, c in MODEL_COLOURS.items()}
    
    df['model'] = df.get('model', None)
    plot_param_vs_metric(
        df,
        param_col='batch_size',
        energy_colours=energy_colours,
        throughput_colours=throughput_colours,
        **kwargs
    )

def plot_num_processes(
    dfs,
    MODEL_COLOURS: Optional[Dict[str, tuple]] = None,
    **kwargs
):
    if isinstance(dfs, dict):
        df = dfs['num_processes'].copy()
    elif isinstance(dfs, pd.DataFrame):
        df = dfs.copy()
    else:
        raise ValueError("dfs must be a dict or a DataFrame")
    
    energy_colours     = {m: c['energy']     for m, c in MODEL_COLOURS.items()}
    throughput_colours = {m: c['throughput'] for m, c in MODEL_COLOURS.items()}
    
    df['model'] = df.get('model', None)
    df['num_processes'] = df['num_processes'].astype(int)
    plot_param_vs_metric(
        df,
        param_col='num_processes',
        energy_colours=energy_colours,
        throughput_colours=throughput_colours,
        **kwargs
    )


def plot_precision(
    dfs,
    MODEL_COLOURS: Optional[Dict[str, tuple]] = None,
    **kwargs
):
    if isinstance(dfs, dict):
        df = dfs.get('precis')
    elif isinstance(dfs, pd.DataFrame):
        df = dfs.copy()
    else:
        raise ValueError("dfs must be a dict or a DataFrame")
    
    energy_colours     = {m: c['energy']     for m, c in MODEL_COLOURS.items()}
    throughput_colours = {m: c['throughput'] for m, c in MODEL_COLOURS.items()}
    
    df = df.copy()
    if 'model' not in df.columns:
        df['model'] = None
    def _mode(r):
        if r.get('load_in_4bit'):      return 'INT4'
        if r.get('load_in_8bit'):      return 'INT8'
        if r.get('fp_precision')=='torch.float16': return 'FP16'
        return 'FP32'
    df['precision'] = pd.Categorical(
        df.apply(_mode, axis=1),
        categories=['FP32','FP16','INT8','INT4'],
        ordered=True
    )
    plot_param_vs_metric(
        df,
        param_col='precision',
        energy_colours=energy_colours,
        throughput_colours=throughput_colours,
        **kwargs
    )
