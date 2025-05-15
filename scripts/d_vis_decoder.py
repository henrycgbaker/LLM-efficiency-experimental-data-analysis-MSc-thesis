import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np

# Helper function unchanged

def _plot_with_band(ax, raw_df, x_col, y_col, mean_df, mean_col, std_col,
                    color, raw_kwargs=None, band_alpha=0.2,
                    line_kwargs=None,
                    normalise_axes=None,
                    plot_mean=True, plot_band=True, plot_raw=True,
                    label_mean=None):
    raw_kwargs = raw_kwargs or {}
    line_kwargs = line_kwargs or {}
    normalise_axes = normalise_axes or []

    # Determine x positions
    idx_str = mean_df.index.astype(str)
    try:
        positions = idx_str.astype(float)
    except ValueError:
        positions = np.arange(len(idx_str))
        ax.set_xticks(positions)
        ax.set_xticklabels(idx_str)
    mapping = {str(v): p for v, p in zip(idx_str, positions)}
    raw_x = raw_df[x_col].astype(str).map(mapping)

    # Compute mean/std
    mean_vals = mean_df[mean_col].values.copy()
    std_vals  = mean_df[std_col].fillna(0).values.copy()
    lower     = mean_vals - std_vals
    upper     = mean_vals + std_vals

    # Normalize if requested
    baseline = None
    is_normalised = False
    if ax in normalise_axes:
        is_normalised = True
        baseline    = mean_vals[0]
        mean_vals  /= baseline
        std_vals   /= baseline
        lower       = mean_vals - std_vals
        upper       = mean_vals + std_vals
        old_label = ax.get_ylabel()
        if "(normalised)" not in old_label:
            ax.set_ylabel(f"{old_label} (normalised)", color=ax.yaxis.label.get_color())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}x"))

    # Scatter raw data
    if plot_raw:
        raw_y = raw_df[y_col] / baseline if baseline is not None else raw_df[y_col]
        ax.scatter(raw_x, raw_y,
                   alpha=raw_kwargs.get('alpha',0.2),
                   marker=raw_kwargs.get('marker','o'),
                   color=raw_kwargs.get('color'),
                   label=None)

    # Plot mean line and band
    x_vals = np.linspace(positions.min(), positions.max(), len(positions)) if len(positions)>1 else positions
    if plot_mean:
        ax.plot(x_vals,
                mean_vals,
                linestyle=line_kwargs.get('linestyle','-'),
                marker=line_kwargs.get('marker',None),
                alpha=line_kwargs.get('alpha',1.0),
                color=line_kwargs.get('color'),
                label=label_mean)
    if plot_band:
        ax.fill_between(x_vals,
                        lower, upper,
                        alpha=band_alpha,
                        color=line_kwargs.get('color'),
                        label=None)

# ---------------------------
# Plot: Decoder Temperature
# ---------------------------

def plot_decoder_temperature(
    dfs,
    normalise_axes=None,
    plot_mean=True,
    plot_band=True,
    plot_raw=True,
    add_baseline_energy=False,
    cycle_id=None,
    model=None
):
    # 1) extract the decoding DataFrame
    if isinstance(dfs, dict):
        df = dfs.get('decoding')
    elif isinstance(dfs, pd.DataFrame):
        df = dfs.copy()
    else:
        raise ValueError("`dfs` must be a dict or pandas DataFrame")
    if df is None:
        raise ValueError("No 'decoding' key found in dfs dict")

    # 2) ensure required cols
    required = {
        'decoder_temperature',
        'energy_per_token_kwh',
        'decoder_config_decoding_mode',
        'model'
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Decoding DataFrame is missing columns: {missing}")

    # 3) default model injection
    if df['model'].isna().all():
        default_label = model if isinstance(model, str) else "Model"
        df['model'] = default_label

    # 4) filters
    if cycle_id is not None:
        df = df[df['cycle_id'] == cycle_id]
    if model is not None:
        models = model if isinstance(model, (list, tuple)) else [model]
        df = df[df['model'].isin(models)]
    else:
        models = sorted(df['model'].unique())

    # 5) setup figure
    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xlabel('Decoder Temperature')
    ax.set_ylabel('Energy-per-Token (kWh)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.grid(True, axis='x', linestyle=':', alpha=0.2)

    norm_axes = [ax] if normalise_axes and 'ax1' in normalise_axes else []

    # 6) loop and plot
    linestyles = ['-', '--', '-.', (0,(5,1))]
    markers = ['o','D','^','s']
    methods = ['greedy','top_k','top_p']
    colors = {'greedy':'tab:blue','top_k':'tab:green','top_p':'tab:red'}

    for i, m_name in enumerate(models):
        style = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)]
        sub_m = df[df['model'] == m_name]

        for method in methods:
            sub = sub_m[sub_m['decoder_config_decoding_mode'] == method]
            if sub.empty:
                continue
            stats = sub.groupby('decoder_temperature').agg(
                energy_mean=('energy_per_token_kwh','mean'),
                energy_std =('energy_per_token_kwh','std')
            )
            _plot_with_band(
                ax, sub, 'decoder_temperature','energy_per_token_kwh',
                stats, 'energy_mean','energy_std',
                color=colors[method],
                raw_kwargs={'alpha':0.2,'marker':marker,'color':colors[method]},
                line_kwargs={'linestyle':style,'marker':marker,'color':colors[method]},
                band_alpha=0.2,
                normalise_axes=norm_axes,
                plot_mean=plot_mean,
                plot_band=plot_band,
                plot_raw=plot_raw,
                label_mean=f"{m_name}:{method}"
            )

    # 7) optional baseline
    if add_baseline_energy:
        for m_name in models:
            base_df = df[(df['model']==m_name) & (df['decoder_config_decoding_mode']=='greedy')]
            if base_df.empty:
                continue
            stats = base_df.groupby('decoder_temperature')['energy_per_token_kwh'].mean()
            base = 1.0 if norm_axes else stats.iloc[0]
            xs = stats.index.astype(float)
            ax.axhline(base, linestyle=':', color='gray', alpha=0.6)
            ax.text(xs[-1], base, f"Baseline {m_name}:greedy",
                    ha='right', va='bottom', fontsize='small', color='gray', alpha=0.4)

    plt.title('Energy-per-Token vs Decoder Temperature')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()


# ---------------------------
# Plot: Decoder Top-k vs Energy
# ---------------------------

def plot_decoder_top_k(
    dfs,
    normalise_axes=None,
    plot_mean=True,
    plot_band=True,
    plot_raw=True,
    add_baseline_energy=False,
    cycle_id=None,
    model=None
):
    # extract + copy
    if isinstance(dfs, dict):
        df = dfs.get('decoding')
    elif isinstance(dfs, pd.DataFrame):
        df = dfs.copy()
    else:
        raise ValueError("`dfs` must be a dict or pandas DataFrame")
    if df is None:
        raise ValueError("No 'decoding' key found in dfs dict")

    # filters
    if cycle_id is not None:
        df = df[df['cycle_id'] == cycle_id]
    if model is not None:
        models = model if isinstance(model, (list, tuple)) else [model]
        df = df[df['model'].isin(models)]
    else:
        models = sorted(df['model'].unique())

    # keep only top_k
    df = df[df['decoder_config_decoding_mode'] == 'top_k']
    if df.empty:
        print("No top_k data to plot.")
        return

    # prepare
    temps = sorted(df['decoder_temperature'].unique())
    cmap = plt.cm.viridis
    colors_t = {t: cmap(i/len(temps)) for i,t in enumerate(temps)}
    linestyles = ['-', '--', '-.', (0,(5,1))]
    markers = ['o','D','^','s']

    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xlabel('Top-k Value')
    ax.set_ylabel('Energy-per-Token (kWh)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.grid(True, axis='x', linestyle=':', alpha=0.2)

    norm_axes = [ax] if normalise_axes and 'ax1' in normalise_axes else []

    for i, m_name in enumerate(models):
        style = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)]
        for t in temps:
            sub = df[(df['model']==m_name) & (df['decoder_temperature']==t)]
            if sub.empty:
                continue
            stats = sub.groupby('decoder_top_k').agg(
                energy_mean=('energy_per_token_kwh','mean'),
                energy_std =('energy_per_token_kwh','std')
            )
            _plot_with_band(
                ax, sub, 'decoder_top_k','energy_per_token_kwh',
                stats, 'energy_mean','energy_std',
                color=colors_t[t],
                raw_kwargs={'alpha':0.2,'marker':marker,'color':colors_t[t]},
                line_kwargs={'linestyle':style,'marker':marker,'color':colors_t[t]},
                band_alpha=0.2,
                normalise_axes=norm_axes,
                plot_mean=plot_mean,
                plot_band=plot_band,
                plot_raw=plot_raw,
                label_mean=f"{m_name}:temp={t}"
            )

    if add_baseline_energy:
        for m_name in models:
            base_df = df[(df['model']==m_name)]
            if base_df.empty:
                continue
            stats = base_df.groupby('decoder_temperature')['energy_per_token_kwh'].mean()
            base = 1.0 if norm_axes else stats.iloc[0]
            xmin, xmax = ax.get_xlim()
            ax.hlines(base, xmin, xmax, linestyle=':', color='gray', alpha=0.4)
            ax.text(xmax, base, f"Baseline {m_name}:greedy",
                    ha='right', va='bottom', fontsize='small', color='gray', alpha=0.4)

    plt.title('Energy-per-Token vs Top-k (grouped by Temperature)')
    ax.legend(loc='best', title='Model:Temp')
    plt.tight_layout()
    plt.show()


# ---------------------------
# Plot: Decoder Top-p vs Energy
# ---------------------------

def plot_decoder_top_p(
    dfs,
    normalise_axes=None,
    plot_mean=True,
    plot_band=True,
    plot_raw=True,
    add_baseline_energy=False,
    cycle_id=None,
    model=None
):
    # extract + copy
    if isinstance(dfs, dict):
        df = dfs.get('decoding')
    elif isinstance(dfs, pd.DataFrame):
        df = dfs.copy()
    else:
        raise ValueError("`dfs` must be a dict or pandas DataFrame")
    if df is None:
        raise ValueError("No 'decoding' key found in dfs dict")

    # filters
    if cycle_id is not None:
        df = df[df['cycle_id'] == cycle_id]
    if model is not None:
        models = model if isinstance(model, (list, tuple)) else [model]
        df = df[df['model'].isin(models)]
    else:
        models = sorted(df['model'].unique())

    # keep only top_p
    df = df[df['decoder_config_decoding_mode'] == 'top_p']
    if df.empty:
        print("No top_p data to plot.")
        return

    # prepare
    temps = sorted(df['decoder_temperature'].unique())
    cmap = plt.cm.viridis
    colors_t = {t: cmap(i/len(temps)) for i,t in enumerate(temps)}
    linestyles = ['-', '--', '-.', (0,(5,1))]
    markers = ['o','D','^','s']

    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xlabel('Top-p Value')
    ax.set_ylabel('Energy-per-Token (kWh)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.grid(True, axis='x', linestyle=':', alpha=0.2)

    norm_axes = [ax] if normalise_axes and 'ax1' in normalise_axes else []

    for i, m_name in enumerate(models):
        style = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)]
        for t in temps:
            sub = df[(df['model']==m_name) & (df['decoder_temperature']==t)]
            if sub.empty:
                continue
            stats = sub.groupby('decoder_top_p').agg(
                energy_mean=('energy_per_token_kwh','mean'),
                energy_std =('energy_per_token_kwh','std')
            )
            _plot_with_band(
                ax, sub, 'decoder_top_p','energy_per_token_kwh',
                stats, 'energy_mean','energy_std',
                color=colors_t[t],
                raw_kwargs={'alpha':0.2,'marker':marker,'color':colors_t[t]},
                line_kwargs={'linestyle':style,'marker':marker,'color':colors_t[t]},
                band_alpha=0.2,
                normalise_axes=norm_axes,
                plot_mean=plot_mean,
                plot_band=plot_band,
                plot_raw=plot_raw,
                label_mean=f"{m_name}:temp={t}"
            )

    if add_baseline_energy:
        for m_name in models:
            base_df = df[(df['model']==m_name)]
            if base_df.empty:
                continue
            stats = base_df.groupby('decoder_temperature')['energy_per_token_kwh'].mean()
            base = 1.0 if norm_axes else stats.iloc[0]
            xmin, xmax = ax.get_xlim()
            ax.hlines(base, xmin, xmax, linestyle=':', color='gray', alpha=0.4)
            ax.text(xmax, base, f"Baseline {m_name}:greedy",
                    ha='right', va='bottom', fontsize='small', color='gray', alpha=0.4)

    plt.title('Energy-per-Token vs Top-p (grouped by Temperature)')
    ax.legend(loc='best', title='Model:Temp')
    plt.tight_layout()
    plt.show()