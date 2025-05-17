from typing import List, Dict
import pandas as pd

import pandas as pd
from typing import List

def get_descriptive_stats(df: pd.DataFrame, models: List[str] = None):
    """
    Compute and print descriptive statistics on the MEAN energy_per_token_kwh per configuration
    for each model, including:
      - max/min scenarios
      - % range relative to mean
      - % range relative to min baseline
      - % energy reduction from worst to best
      - variability & distribution metrics
    Then repeat all “range” stats with outliers removed (1.5×IQR method).
    """
    available_models = df['model'].unique()
    models = [m for m in (models or available_models) if m in available_models]

    print(f"\nModels: {models}")

    for model in models:
        model_df = df[df['model'] == model]
        if model_df.empty:
            print(f"No data for model {model}, skipping.")
            continue

        def summarize(sub_df: pd.DataFrame, label: str):
            # 1) aggregate to mean per config
            grouped = (
                sub_df
                .groupby('config_name')['energy_per_token_kwh']
                .mean()
                .reset_index(name='mean_energy')
            )

            # 2) max/min
            idx_max = grouped['mean_energy'].idxmax()
            idx_min = grouped['mean_energy'].idxmin()
            max_val = grouped.loc[idx_max, 'mean_energy']
            min_val = grouped.loc[idx_min, 'mean_energy']

            # 3) percent energy reduction baseline→best
            pct_reduction = (max_val - min_val) / max_val

            # 4) relative to mean of means
            overall_mean = grouped['mean_energy'].mean()
            range_rel_mean = (max_val - min_val) / overall_mean

            # 5) normalized to min baseline
            range_rel_min = (max_val - min_val) / min_val
            grouped['norm_to_min'] = grouped['mean_energy'] / min_val
            grouped['diff_to_min_pct'] = (grouped['mean_energy'] - min_val) / min_val

            print(f"\n--- {model} ({label}) ---")
            print(f"Max mean energy: {max_val:.4f} kWh @ {grouped.loc[idx_max, 'config_name']}")
            print(f"Min mean energy: {min_val:.4f} kWh @ {grouped.loc[idx_min, 'config_name']}")
            print(f"- Energy reduction (worst→best): {pct_reduction:.2%}")
            print(f"- Range vs. mean of means: {range_rel_mean:.2%}")
            print(f"- Range vs. min baseline: {range_rel_min:.2%}")
            print("\nNormalized to min baseline:")
            print(grouped[['config_name', 'norm_to_min', 'diff_to_min_pct']].to_string(index=False))

            # variability & distribution metrics
            std_means = grouped['mean_energy'].std()
            cv = std_means / overall_mean
            q1 = grouped['mean_energy'].quantile(0.25)
            median = grouped['mean_energy'].median()
            q3 = grouped['mean_energy'].quantile(0.75)
            iqr = q3 - q1
            skew = grouped['mean_energy'].skew()
            kurt = grouped['mean_energy'].kurtosis()
            count = grouped.shape[0]

            print(f"\nVariability & distribution:")
            print(f" Count: {count}")
            print(f" Std dev: {std_means:.4f} kWh ({cv:.2%} of mean)")
            print(f" Quartiles (25% / 50% / 75%): {q1:.4f}, {median:.4f}, {q3:.4f}")
            print(f" IQR: {iqr:.4f}")
            print(f" Skewness: {skew:.2f}, Kurtosis: {kurt:.2f}")
            print("----")

        # 1️⃣ Summaries on the raw data
        summarize(model_df, label="raw data")

        # 2️⃣ Remove outliers (1.5×IQR on the raw energy values) and re-run
        q1_all = model_df['energy_per_token_kwh'].quantile(0.25)
        q3_all = model_df['energy_per_token_kwh'].quantile(0.75)
        iqr_all = q3_all - q1_all
        lower = q1_all - 1.5 * iqr_all
        upper = q3_all + 1.5 * iqr_all
        cleaned = model_df[
            model_df['energy_per_token_kwh'].between(lower, upper)
        ]

        summarize(cleaned, label="outliers removed (1.5×IQR)")



def compare_energy_to_appliances(
    df,
    avg_len_tokens: int = 300,
    appliances_kwh: Dict[str, float] = None,
    models: List[str] = None
):
    """
    Scenarios:
    1) Full config means
    2) Config means without outliers
    3) By config_name details
    4) Groups: Realistic vs Artificial configs
    5) Comparison: Realistic vs Artificial group means
    """
    # Default appliance energy usages
    if appliances_kwh is None:        
        appliances_kwh = {
            "iPhone_charge":     0.015,    # kWh per full charge
            "MacBook_charge":    0.08,     # kWh per full charge
            "wifi_router_24h":   0.24,     # kWh per 24 h
            "streaming_1hr":     0.6,      # kWh per hour streaming
            "google_search":     0.0003,   # kWh per search
            "kettle":            0.075,    # kWh per boil (~3 min)
            "shower":            1.58      # kWh per 10 min electric shower
        }


    # Determine models to include
    available = df['model'].unique()
    models = [m for m in (models or available) if m in available]

    print(f"== ASSUMING AVERAGE LENGTH: {avg_len_tokens} TOKENS ==")
    print(f"Models: {models}\n")

    for model in models:
        print(f"=== Model: {model} ===")
        mdf = df[df['model'] == model]
        if mdf.empty:
            print("No data for this model.\n")
            continue

        # Compute mean energy per config
        config_stats = (
            mdf
            .groupby('config_name')['energy_per_token_kwh']
            .mean()
            .reset_index(name='mean_energy')
        )
        # Add per-response energy
        config_stats['response_energy'] = config_stats['mean_energy'] * avg_len_tokens

        # Scenario 1 & 2 on config means
        mean_vals = config_stats['response_energy']
        std_vals = mean_vals.std()
        full_cfg = config_stats.copy()
        clean_cfg = config_stats[abs(mean_vals - mean_vals.mean()) <= 3 * std_vals]

        for label, subset in [("Full config means", full_cfg), ("Without outlier configs", clean_cfg)]:
            n_cfg = len(subset)
            e_max = subset['response_energy'].max()
            e_min = subset['response_energy'].min()
            e_mean = subset['response_energy'].mean()
            diff = e_max - e_min
            ratio = e_max / e_min if e_min > 0 else float('inf')

            print(f"-- Scenario: {label} ({n_cfg} configs) --")
            print(f"Overall ratio (max/min): {ratio:.2f}")
            print("# responses to match appliance (worst/best/diff/mean):")
            for app, kwh in appliances_kwh.items():
                wc = kwh / e_max if e_max > 0 else float('inf')
                bc = kwh / e_min if e_min > 0 else float('inf')
                dc = kwh / diff if diff > 0 else float('inf')
                mc = kwh / e_mean if e_mean > 0 else float('inf')
                print(f"    {app}: worst {wc:.2f}, best {bc:.2f}, diff {dc:.2f}, mean {mc:.2f}")
            print()

        # Scenario 3: Detailed per-config responses
        print("-- Scenario: By config_name details --")
        for _, row in config_stats.iterrows():
            resp_e = row['response_energy']
            print(f"Config: {row['config_name']} → {resp_e:.5f} kWh per response")
            for app, kwh in appliances_kwh.items():
                print(f"    {app}: {kwh / resp_e:.2f} responses")
        print()

        # Scenario 4: Group summaries (Realistic vs Artificial)
        groups = {
            'Realistic': config_stats[config_stats['config_name'].str.startswith('R')],
            'Artificial': config_stats[config_stats['config_name'].str.startswith('A')]
        }
        for label, grp in groups.items():
            m_energy = grp['response_energy'].mean() if not grp.empty else 0
            n_cfg = len(grp)
            print(f"-- Scenario: Group {label} ({n_cfg} configs) --")
            print(f"Mean kWh per response: {m_energy:.5f}")
            for app, kwh in appliances_kwh.items():
                val = kwh / m_energy if m_energy > 0 else float('inf')
                print(f"    {app}: {val:.2f} responses")
            print()

        # Scenario 5: Compare group means
        e_real = groups['Realistic']['response_energy'].mean() if not groups['Realistic'].empty else 0
        e_art = groups['Artificial']['response_energy'].mean() if not groups['Artificial'].empty else 0
        diff_ga = e_real - e_art
        ratio_ga = e_real / e_art if e_art > 0 else float('inf')
        print("-- Scenario: Realistic vs Artificial --")
        print(f"Realistic mean: {e_real:.5f} kWh, Artificial mean: {e_art:.5f} kWh")
        print(f"Difference: {diff_ga:.5f} kWh, Ratio: {ratio_ga:.2f}x")
        for app, kwh in appliances_kwh.items():
            real_c = kwh / e_real if e_real > 0 else float('inf')
            art_c = kwh / e_art if e_art > 0 else float('inf')
            diff_c = kwh / diff_ga if diff_ga > 0 else float('inf')
            print(f"    {app}: Realistic {real_c:.2f}, Artificial {art_c:.2f}, Diff {diff_c:.2f}")
        print()
    
def artificial_v_realistic(
    df: pd.DataFrame,
    avg_len_tokens: int = 300,
    models: List[str] = None
):
    """
    For each model:
      - Compute mean kWh per response for Realistic vs Artificial configs
      - Print absolute diff, ratio, % reduction
    Do it twice: on raw data, and after removing outliers (1.5×IQR on energy_per_token_kwh).
    """
    available = df['model'].unique()
    models = [m for m in (models or available) if m in available]

    for model in models:
        mdf = df[df['model'] == model]
        if mdf.empty:
            print(f"Model {model}: no data, skipping.")
            continue

        def summary(sub_df: pd.DataFrame, label: str):
            # compute per-config mean & per-response energy
            cfg = (
                sub_df
                .groupby('config_name')['energy_per_token_kwh']
                .mean()
                .reset_index(name='mean_energy')
            )
            cfg['response_energy'] = cfg['mean_energy'] * avg_len_tokens

            real = cfg[cfg['config_name'].str.startswith('R')]
            art  = cfg[cfg['config_name'].str.startswith('A')]

            e_real = real['response_energy'].mean() if not real.empty else 0.0
            e_art  = art['response_energy'].mean()  if not art.empty  else 0.0

            diff      = e_art - e_real
            ratio     = (e_real / e_art) if e_art > 0 else float('nan')
            reduction = (e_art - e_real) / e_art if e_art > 0 else float('nan')

            print(f"\n--- {model} ({label}) ---")
            print(f"Realistic mean   : {e_real:.5f} kWh/resp")
            print(f"Artificial mean  : {e_art:.5f} kWh/resp")
            print(f"- Abs diff (A−R) : {diff:.5f} kWh")
            print(f"- Ratio (R/A)    : {ratio:.2f}×")
            print(f"- % reduction    : {reduction:.2%}")

        # raw data
        summary(mdf, "raw data")

        # remove outliers on token-level energy before grouping
        q1 = mdf['energy_per_token_kwh'].quantile(0.25)
        q3 = mdf['energy_per_token_kwh'].quantile(0.75)
        iqr = q3 - q1
        cleaned = mdf[
            mdf['energy_per_token_kwh'].between(q1 - 1.5*iqr, q3 + 1.5*iqr)
        ]

        # without outliers
        summary(cleaned, "outliers removed (1.5×IQR)")


def within_realistic(
    df: pd.DataFrame,
    avg_len_tokens: int = 300,
    models: List[str] = None
):
    """
    For each model, among Realistic configs only:
      - Print worst vs best response-energy, abs diff, % reduction
    Do it twice: on raw data, and after removing outliers (1.5×IQR on energy_per_token_kwh).
    """
    available = df['model'].unique()
    models = [m for m in (models or available) if m in available]

    for model in models:
        mdf = df[df['model'] == model]
        if mdf.empty:
            print(f"Model {model}: no data, skipping.")
            continue

        def summary(sub_df: pd.DataFrame, label: str):
            real = (
                sub_df[sub_df['config_name'].str.startswith('R')]
                .groupby('config_name')['energy_per_token_kwh']
                .mean()
                .reset_index(name='mean_energy')
            )
            if real.empty:
                print(f"{model} ({label}): no realistic configs.")
                return

            real['response_energy'] = real['mean_energy'] * avg_len_tokens
            idx_max = real['response_energy'].idxmax()
            idx_min = real['response_energy'].idxmin()
            e_max = real.loc[idx_max, 'response_energy']
            e_min = real.loc[idx_min, 'response_energy']

            diff = e_max - e_min
            pct_reduction = (e_max - e_min) / e_max if e_max > 0 else float('nan')

            print(f"\n*** {model} (Realistic — {label}) ***")
            print(f"Worst (max): {real.loc[idx_max, 'config_name']} @ {e_max:.5f} kWh")
            print(f"Best  (min): {real.loc[idx_min, 'config_name']} @ {e_min:.5f} kWh")
            print(f"- Abs diff  : {diff:.5f} kWh")
            print(f"- % reduction: {pct_reduction:.2%}")

        # raw data
        summary(mdf, "raw data")

        # remove outliers on token-level energy before grouping
        q1 = mdf['energy_per_token_kwh'].quantile(0.25)
        q3 = mdf['energy_per_token_kwh'].quantile(0.75)
        iqr = q3 - q1
        cleaned = mdf[
            mdf['energy_per_token_kwh'].between(q1 - 1.5*iqr, q3 + 1.5*iqr)
        ]

        # without outliers
        summary(cleaned, "outliers removed (1.5×IQR)")
