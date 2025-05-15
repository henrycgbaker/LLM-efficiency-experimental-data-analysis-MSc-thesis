import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib import cm
import numpy as np
from pandas.api.types import is_numeric_dtype

# -----------------------------------------------------------------
# Helpers for coloring by model
# -----------------------------------------------------------------
# unused:
def assign_model_colors(df):
    """
    Create a unique color mapping for each model using a Tab10 colormap.
    """
    models = df['model'].unique()
    cmap = cm.get_cmap('tab10', len(models))
    return {m: cmap(i) for i, m in enumerate(models)}

# -----------------------------------------------------------------
# Plot: Energy Histogram overlayed by Model
# -----------------------------------------------------------------
def plot_energy_histograms(df, model_colours, bins=90):
    """
    Overlay histograms of energy_per_token_kwh for all models on one plot.
    """
    plt.figure(figsize=(8, 6))
    for model, subset in df.groupby('model'):
        plt.hist(subset['energy_per_token_kwh'], 
                 bins=bins,
                 alpha=0.5, 
                 label=model, 
                 color=model_colours[model]['energy'])
    plt.xlabel('Energy per Token (kWh)')
    plt.ylabel('Count')
    plt.title('Histogram of Energy per Token by Model')
    plt.legend(title='Model')
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------
# Plot: Boxplot of Energy per Token overlayed by Model
# -----------------------------------------------------------------
def plot_energy_boxplots(df, model_colours, column='energy_per_token_kwh'):
    """
    Overlay horizontal boxplots of the specified column for all models on one plot.
    """
    if column not in df:
        print(f"Column '{column}' not found in DataFrame.")
        return
    column_title = 'Energy per Token (kWh)'
    models = list(model_colours)
    data = [df[df['model'] == m][column].dropna() for m in models]
    fig, ax = plt.subplots(figsize=(10, 2 + len(models)*0.3))
    box = ax.boxplot(data, vert=False, patch_artist=True, labels=models)
    for patch, model in zip(box['boxes'], models):
        patch.set_facecolor(model_colours[model]['energy'])
        patch.set_alpha(0.6)
    ax.set_xlabel(column_title)
    ax.set_title(f'Boxplot of {column_title} by Model')
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------
# Plot: Throughput vs Energy overlayed scatter by Model
# -----------------------------------------------------------------
def plot_throughput_vs_energy(df, model_colours):
    """
    Overlay scatter of throughput_queries_per_sec vs energy_per_token_kwh for all models.
    """
    plt.figure(figsize=(8, 6))
    for model, subset in df.groupby('model'):
        plt.scatter(subset['throughput_queries_per_sec'],
                    subset['energy_per_token_kwh'],
                    alpha=0.7, label=model, color=model_colours[model]['energy'])
    plt.xlabel('Throughput (queries/sec)')
    plt.ylabel('Energy per Token (kWh)')
    plt.title('Throughput vs Energy per Token by Model')
    plt.legend(title='Model')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------
# Plot: General Scatter overlayed by Model
# -----------------------------------------------------------------
def plot_scatter_by_model(df, model_colours, x_column, y_column):
    """
    Overlay scatter of y_column vs x_column for all models.
    """
    if x_column not in df or y_column not in df:
        print(f"Columns '{x_column}' or '{y_column}' not found in DataFrame.")
        return
    plt.figure(figsize=(8, 6))
    for model, subset in df.groupby('model'):
        plt.scatter(subset[x_column], subset[y_column],
                    alpha=0.7, label=model, color=model_colours[model]['energy'])
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'{y_column} vs {x_column} by Model')
    plt.legend(title='Model')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------
# Plot: Correlation Matrix per Model
# -----------------------------------------------------------------
def plot_corr_by_model(df):

    cols = [
        'energy_per_token_kwh',
        #'flops_per_token',
        'average_latency_ms_per_batch',
        #'throughput_queries_per_sec',
        'throughput_tokens_per_sec',
        'gpu_utilization_proc_all',
        'gpu_power_proc_all'
    ]
    for model, subset in df.groupby('model'):
        data = subset[cols].dropna()
        if data.shape[0] < 2:
            continue
        corr = data.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr, vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(cols)))
        ax.set_yticks(np.arange(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha='right')
        ax.set_yticklabels(cols)
        ax.set_title(f'Correlation Matrix for {model}')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Correlation coeff', rotation=-90, va='bottom')
        plt.tight_layout()
        plt.show()

# -----------------------------------------------------------------
# Plot: Divergence vs Various X columns overlayed by Model
# -----------------------------------------------------------------
def plot_divergence_by_model(df, model_colours, x_column):
    """
    Overlay scatter of divergence_energy_flops vs x_column for all models.
    """
    if x_column not in df:
        print(f"Column '{x_column}' not found in DataFrame.")
        return
    plt.figure(figsize=(8, 6))
    for model, subset in df.groupby('model'):
        plt.scatter(subset[x_column], subset['divergence_energy_flops'],
                    alpha=0.7, label=model, color=model_colours[model]['energy'])
    plt.xlabel(x_column)
    plt.ylabel('divergence_energy_flops')
    plt.title(f'Divergence vs {x_column} by Model')
    plt.legend(title='Model')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------
# Master function to run all model-specific plots
# -----------------------------------------------------------------
def plot_all_diagnostics(df, df_1b=None, df_3b=None, MODEL_COLOURS=None):

    model_colours = MODEL_COLOURS
    
    # Histograms and boxplots
    print("📊 Plotting histogram...")
    plot_energy_histograms(df, model_colours)

    print("📦 Plotting boxplot...")
    plot_energy_boxplots(df, model_colours)
    
    print("📊 Plotting individual histogram...")
    plot_energy_histograms(df_1b, model_colours)
    plot_energy_histograms(df_3b, model_colours)
    
    # Scatter plots
    print("🔬 Scatter: by model...")
    plot_scatter_by_model(df, model_colours, 'flops_per_token', 'energy_per_token_kwh')

    # Correlations
    print("🔗 Correlation matrix...")
    plot_corr_by_model(df)

    print("🔬 Scatter: Throughput vs energy...")
    plot_throughput_vs_energy(df, model_colours)

    # Divergence comparisons
    print("📈 Scatter: Divergence patterns...")
    x_vars = [
        'num_processes',
        'batch_size___fixed_batching',
        'decoder_config_decoding_mode' if 'decoder_config_decoding_mode' in df.columns else None,
        'temperature' if 'temperature' in df.columns else None,
        'latency_numeric' if 'latency_numeric' in df.columns else None
    ]
    for x in filter(None, x_vars):
        if not is_numeric_dtype(df[x]):
            print(f"Skipping non-numeric column {x!r}")
            continue
        plot_divergence_by_model(df, model_colours, x)

# ---------------------------
# Main Entry Point
# ---------------------------

if __name__ == "__main__":

    from scripts.a_data_loading_cleaning import run_load_clean_diagnose_data

    csv_path = "results/controlled_results.csv"
    df = run_load_clean_diagnose_data(csv_path)
    
    plot_all_diagnostics(df, df_1b, df_3b)