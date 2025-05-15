import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from matplotlib import cm

# — original helpers —
def add_latency_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['latency_numeric'] = df.apply(
        lambda r: 0.0 if not r['latency_simulation_simulate']
                  else (r['latency_simulation_delay_min'] +
                        r['latency_simulation_delay_max']) / 2,
        axis=1
    )
    return df

def classify_by_burst_size_and_constant(df: pd.DataFrame):
    df2 = add_latency_numeric(df.copy())
    origin = df2['latency_numeric'] == 0.0
    df2['class_const'] = (
        (df2['latency_simulation_simulate'] & ~df2['latency_simulation_simulate_burst'])
        | origin
    )
    burst_rows = df2['latency_simulation_simulate_burst']
    sizes = sorted(df2.loc[burst_rows, 'latency_simulation_burst_size'].unique())
    for sz in sizes:
        col = f'class_burst_{sz}'
        df2[col] = (
            (df2['latency_simulation_simulate_burst'] &
             (df2['latency_simulation_burst_size'] == sz))
            | origin
        )
    return df2, sizes

def _plot_with_band(ax, raw_df, x_col, y_col, mean_df, mean_col, std_col,
                    color, raw_kwargs=None, band_alpha=0.2,
                    line_kwargs=None,
                    normalise_axes=None,
                    plot_mean=True, plot_band=True, plot_raw=True,
                    label_mean=None):
    raw_kwargs   = raw_kwargs or {}
    line_kwargs  = line_kwargs or {}
    normalise_axes = normalise_axes or []

    # map string index → numeric positions
    idx = mean_df.index.astype(str)
    try:
        pos = idx.astype(float)
    except ValueError:
        pos = np.arange(len(idx))
        ax.set_xticks(pos)
        ax.set_xticklabels(idx)
    mapper = {s:p for s,p in zip(idx, pos)}
    raw_x = raw_df[x_col].astype(str).map(mapper)

    # mean & std arrays
    mean_vals = mean_df[mean_col].values.astype(float)
    std_vals  = mean_df[std_col].fillna(0).values.astype(float)
    lower     = mean_vals - std_vals
    upper     = mean_vals + std_vals

    # normalise if requested
    if ax in normalise_axes:
        baseline = mean_vals[0]
        mean_vals /= baseline
        lower     = (mean_vals - std_vals/baseline)
        upper     = (mean_vals + std_vals/baseline)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.2f}x"))
        old_ylabel = ax.get_ylabel()
        if "(normalized)" not in old_ylabel:
            ax.set_ylabel(f"{old_ylabel} (normalized)", color=ax.yaxis.label.get_color())

    # raw scatter
    if plot_raw and not raw_df.empty:
        y = raw_df[y_col].values.astype(float)
        if ax in normalise_axes:
            y /= raw_vals[0]  # same baseline
        ax.scatter(raw_x, y, **raw_kwargs)

    # mean line
    if plot_mean and not mean_df.empty:
        ax.plot(pos, mean_vals, **line_kwargs, label=label_mean)

    # ±1 std band
    if plot_band and not mean_df.empty:
        ax.fill_between(pos, lower, upper,
                        color=line_kwargs.get('color'),
                        alpha=band_alpha)


def plot_latency_burst_size(
    dfs,
    normalise_axes=None,
    plot_mean=True,
    plot_band=True,
    plot_raw=True,
    add_baseline_energy=False,
    cycle_id=None,
    model=None
):
    if isinstance(dfs, dict):
        df = dfs.get('latency')
    elif isinstance(dfs, pd.DataFrame):
        df = dfs
    else:
        print("Invalid input: dfs should be a DataFrame or a dictionary.")
        return

    # filter cycle and models
    if cycle_id is not None:
        df = df[df['cycle_id']==cycle_id]
    if model is not None:
        models = model if isinstance(model, (list,tuple)) else [model]
        df = df[df['model'].isin(models)]
    else:
        models = sorted(df['model'].unique())

    # styling arrays
    linestyles = ['-', '--', '-.', (0,(5,1))]
    markers    = ['o','D','^','s']
    a_line, a_scatter, a_band = 1.0, 0.2, 0.2

    # classify and get burst‐sizes
    df2, sizes = classify_by_burst_size_and_constant(df)

    # build (label, col, color)
    classes = [('Constant', 'class_const', 'tab:orange')]
    cmap = cm.get_cmap('Blues', len(sizes)+1)
    for i, sz in enumerate(sizes, start=1):
        classes.append((f'Burst {sz}', f'class_burst_{sz}', cmap(i)))

    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Energy-per-Token (kWh)')
    ax.set_title('Energy-per-Token vs Latency (grouped by Burst Size)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.grid(True, axis='x', linestyle=':',  alpha=0.2)

    norm_axes = [ax] if normalise_axes and 'ax1' in normalise_axes else []

    # loop models × classes
    for mi, m in enumerate(models):
        ls = linestyles[mi % len(linestyles)]
        mk = markers[mi % len(markers)]
        for label, col, color in classes:
            sub   = df2[(df2['model']==m) & (df2[col])]
            if sub.empty: 
                continue
            stats = sub.groupby('latency_numeric').agg(
                energy_mean=('energy_per_token_kwh','mean'),
                energy_std =('energy_per_token_kwh','std')
            )
            _plot_with_band(
                ax, sub, 'latency_numeric','energy_per_token_kwh',
                stats, 'energy_mean','energy_std',
                color=color,
                raw_kwargs = {'alpha':a_scatter,'marker':mk,'color':color},
                line_kwargs={'linestyle':ls,'marker':mk,'alpha':a_line,'color':color},
                band_alpha = a_band,
                normalise_axes = norm_axes,
                plot_mean = plot_mean,
                plot_band = plot_band,
                plot_raw = plot_raw,
                label_mean = f"{m}:{label}"
            )

    # optional baselines
    if add_baseline_energy:
        for m in models:
            bdf = df[(df['model']==m)&(df['decoder_config_decoding_mode']=='greedy')]
            if bdf.empty: continue
            stats = bdf.groupby('decoder_temperature')['energy_per_token_kwh'].mean()
            base  = 1.0 if ax in norm_axes else stats.iloc[0]
            xs    = stats.index.astype(float)
            lbl   = f"Baseline {m}:greedy"
            ax.axhline(base, linestyle=':', color='gray', alpha=0.6)
            ax.text(xs[-1], base, lbl, ha='right', va='bottom',
                    fontsize='small', color='gray', alpha=0.4)

    # ── custom x-tick mapping: position=mean, label="min-max" ──
    all_means = sorted(df2['latency_numeric'].unique())
    tick_labels = []
    for μ in all_means:
        grp = df2[df2['latency_numeric']==μ]
        lo = grp['latency_simulation_delay_min'].min()
        hi = grp['latency_simulation_delay_max'].max()
        tick_labels.append(f"between \n{lo} - {hi}")
    ax.set_xticks(all_means)
    ax.set_xticklabels(tick_labels)

    ax.legend(loc='best', title='Model:Class')
    plt.tight_layout()
    plt.show()    

def classify_by_burst_interval_and_constant(df: pd.DataFrame):
    df2 = add_latency_numeric(df.copy())
    origin = df2['latency_numeric'] == 0.0
    df2['class_const'] = (
        (df2['latency_simulation_simulate'] & ~df2['latency_simulation_simulate_burst'])
        | origin
    )
    burst_rows = df2['latency_simulation_simulate_burst']
    intervals = sorted(df2.loc[burst_rows, 'latency_simulation_burst_interval'].unique())
    for iv in intervals:
        col = f'class_interval_{iv}'
        df2[col] = (
            (df2['latency_simulation_simulate_burst'] &
             (df2['latency_simulation_burst_interval'] == iv))
            | origin
        )
    return df2, intervals

def plot_latency_burst_interval(
    dfs,
    normalise_axes=None,
    plot_mean=True,
    plot_band=True,
    plot_raw=True,
    add_baseline_energy=False,
    cycle_id=None,
    model=None
):
    if isinstance(dfs, dict):
        df = dfs.get('latency')
    elif isinstance(dfs, pd.DataFrame):
        df = dfs
    else:
        print("Invalid input: dfs should be a DataFrame or a dictionary.")
        return

    # filter cycle and models
    if cycle_id is not None:
        df = df[df['cycle_id']==cycle_id]
    if model is not None:
        models = model if isinstance(model, (list,tuple)) else [model]
        df = df[df['model'].isin(models)]
    else:
        models = sorted(df['model'].unique())

    linestyles = ['-', '--', '-.', (0,(5,1))]
    markers    = ['o','D','^','s']
    a_line, a_scatter, a_band = 1.0, 0.2, 0.2

    # classify by burst interval
    df2, intervals = classify_by_burst_interval_and_constant(df)

    # build classes
    classes = [('Constant', 'class_const', 'tab:orange')]
    cmap = cm.get_cmap('Greens', len(intervals)+1)
    for i, iv in enumerate(intervals, start=1):
        classes.append((f'Interval {iv}', f'class_interval_{iv}', cmap(i)))

    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Energy-per-Token (kWh)')
    ax.set_title('Energy-per-Token vs Latency (grouped by Burst Interval)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.grid(True, axis='x', linestyle=':',  alpha=0.2)

    norm_axes = [ax] if normalise_axes and 'ax1' in normalise_axes else []

    for mi, m in enumerate(models):
        ls = linestyles[mi % len(linestyles)]
        mk = markers[mi % len(markers)]
        for label, col, color in classes:
            sub   = df2[(df2['model']==m) & (df2[col])]
            if sub.empty:
                continue
            stats = sub.groupby('latency_numeric').agg(
                energy_mean=('energy_per_token_kwh','mean'),
                energy_std =('energy_per_token_kwh','std')
            )
            _plot_with_band(
                ax, sub, 'latency_numeric','energy_per_token_kwh',
                stats, 'energy_mean','energy_std',
                color=color,
                raw_kwargs = {'alpha':a_scatter,'marker':mk,'color':color},
                line_kwargs={'linestyle':ls,'marker':mk,'alpha':a_line,'color':color},
                band_alpha = a_band,
                normalise_axes = norm_axes,
                plot_mean = plot_mean,
                plot_band = plot_band,
                plot_raw = plot_raw,
                label_mean = f"{m}:{label}"
            )

    if add_baseline_energy:
        for m in models:
            bdf = df[(df['model']==m)&(df['decoder_config_decoding_mode']=='greedy')]
            if bdf.empty: continue
            stats = bdf.groupby('decoder_temperature')['energy_per_token_kwh'].mean()
            base  = 1.0 if ax in norm_axes else stats.iloc[0]
            xs    = stats.index.astype(float)
            lbl   = f"Baseline {m}:greedy"
            ax.axhline(base, linestyle=':', color='gray', alpha=0.6)
            ax.text(xs[-1], base, lbl, ha='right', va='bottom',
                    fontsize='small', color='gray', alpha=0.4)

    # ── custom x-tick mapping: position=mean, label="min-max" ──
    all_means = sorted(df2['latency_numeric'].unique())
    tick_labels = []
    for μ in all_means:
        grp = df2[df2['latency_numeric']==μ]
        lo = grp['latency_simulation_delay_min'].min()
        hi = grp['latency_simulation_delay_max'].max()
        tick_labels.append(f"between \n{lo} - {hi}")
    ax.set_xticks(all_means)
    ax.set_xticklabels(tick_labels)

    ax.legend(loc='best', title='Model:Class')
    plt.tight_layout()
    plt.show()