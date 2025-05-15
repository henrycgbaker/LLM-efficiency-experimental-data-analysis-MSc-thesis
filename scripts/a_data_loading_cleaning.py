import pandas as pd
import re
import warnings
from typing import (
    Union,
    Sequence,
    Optional
)

# -----------------------------------------------------------------
# lOAD & CLEAN
# Define helper functions to clean column names and resolve duplicates
# -----------------------------------------------------------------
def clean_column(col: str) -> str:
    """
    Cleans an individual column name:
      - Strips whitespace and replaces non-standard quotes,
      - Applies special column name mappings,
      - Removes the 'variables_' prefix,
      - Simplifies per-process column names using regex patterns.
    """
    col = col.strip()
    col = col.replace("“", "\"").replace("”", "\"")
    
    # Special mappings for some columns.
    special_mappings = {
        "setup_cpu_model": "cpu_model",
        "setup_gpu_model": "gpu_model",
        "setup_cycle_id": "cycle_id",
        "model_architecture_total_params": "total_params",
        "model_architecture_architecture": "model_arch"
    }
    if col in special_mappings:
        return special_mappings[col]
    
    # Remove the 'variables_' prefix if it exists.
    if col.startswith("variables_"):
        col = col[len("variables_"):]
    
    # Check for per-process metric patterns (e.g., cpu_power_process_0)
    per_process_patterns = [
        r'(cpu_power_process_\d+)',
        r'(gpu_power_process_\d+)',
        r'(ram_power_process_\d+)',
        r'(cpu_energy_process_\d+)',
        r'(gpu_energy_process_\d+)',
        r'(ram_energy_process_\d+)',
        r'(total_energy_kwh_process_\d+)',
        r'(total_energy_joules_process_\d+)'
    ]
    for pattern in per_process_patterns:
        match = re.search(pattern, col)
        if match:
            return match.group(1)
    
    # For non-per-process columns, search for a known token in the cleaned column.
    tokens = [
        "config_name", "experiment_id", "date_time", "model", "is_encoder_decoder",
        "task_type", "available_gpu_count", "gpu_model", "available_cpu_count", "cpu_model",
        "os", "python_version", "country", "region", "fsdp_use_orig_params", "fsdp_cpu_offload",
        "sharding_strategy", "distributed_type", "num_processes", "max_input_tokens", "max_output_tokens",
        "number_input_prompts", "decode_token_to_text", "decoder_temperature", "decoder_top_k", "decoder_top_p",
        "query_rate", "latency_simulate", "latency_delay_min", "latency_delay_max", "latency_simulate_burst",
        "latency_burst_interval", "latency_burst_size", "fp_precision", "quantization", "load_in_8bit",
        "load_in_4bit", "cached_flops_for_quantised_models", "batch_size___fixed_batching", "adaptive_batching",
        "adaptive_max_tokens", "max_batch_size___adaptive_batching", "inference_type", "backend", "total_params",
        "architecture", "total_input_tokens", "total_generated_tokens", "total_inference_time_sec", 
        "average_latency_ms_per_batch", "throughput_queries_per_sec", "throughput_tokens_per_sec", "flops",
        "gpu_current_memory_allocated_bytes", "gpu_max_memory_allocated_bytes", "gpu_current_memory_reserved_bytes",
        "gpu_max_memory_reserved_bytes", "gpu_utilization_percent", "cpu_usage_percent", "cpu_memory_usage_bytes",
        # Per-process metrics:
        "cpu_power_process_0", "gpu_power_process_0", "ram_power_process_0",
        "cpu_energy_process_0", "gpu_energy_process_0", "ram_energy_process_0",
        "total_energy_kwh_process_0", "total_energy_joules_process_0",
        # Global averages and totals:
        "cpu_power_avg", "gpu_power_avg", "ram_power_avg", "cpu_energy_total", "gpu_energy_total", "ram_energy_total",
        "total_energy_kwh", "total_energy_joules", "tokens_per_joule", "joules_per_token", "flops_per_joule", "joules_per_flop",
        "per-process_emissions"
    ]
    for token in tokens:
        if token in col:
            idx = col.find(token)
            return col[idx:]
    
    return col

def resolve_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    For columns that appear multiple times, this function selects only one instance.
    In special cases (e.g., 'adaptive_batching'), it chooses the column with the desired dtype.
    """
    seen = {}
    for idx, col in enumerate(df.columns):
        seen.setdefault(col, []).append(idx)
    
    chosen_indices = []
    for col, indices in seen.items():
        if len(indices) == 1:
            chosen_indices.append(indices[0])
        else:
            # Example special handling for boolean columns (like adaptive_batching)
            if col == "adaptive_batching":
                bool_idx = None
                for i in indices:
                    if pd.api.types.is_bool_dtype(df.iloc[:, i]):
                        bool_idx = i
                        break
                chosen_indices.append(bool_idx if bool_idx is not None else indices[0])
            else:
                # Otherwise, just keep the first occurrence.
                chosen_indices.append(indices[0])
    
    chosen_indices.sort()  # keep original order as much as possible
    return df.iloc[:, chosen_indices]

def clean_and_reorder_columns(df: pd.DataFrame, desired_order: list) -> pd.DataFrame:
    """
    Cleans the column names and then reorders the DataFrame columns according to `desired_order`.
    Any columns not specified in the order list are appended afterwards.
    """
    mapping = {col: clean_column(col) for col in df.columns}
    df = df.rename(columns=mapping)
    df = resolve_duplicates(df)
    
    # Order columns: first those in desired_order, then the remaining.
    ordered_cols = [col for col in desired_order if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in desired_order]
    final_order = ordered_cols + remaining_cols
    return df[final_order]

# Define preferred column order 
desired_order = [
    "config_name",
    "experiment_id",
    "cycle_id",
    "date_time",
    "model",
    # num_process
    "num_processes",
    # batching
    "batch_size___fixed_batching",
    # decoder parameters
    "decoder_temperature",
    "decoder_top_k",
    "decoder_top_p",
    # latency
    "latency_simulation_simulate",
    "latency_simulation_delay_max",
    "latency_simulation_delay_min",
    "latency_simulation_simulate_burst",
    "latency_simulation_burst_size",
    "latency_simulation_burst_interval",
    # precision / quantisation
    "fp_precision",
    "quantization",
    "load_in_8bit",
    "load_in_4bit",
    "cached_flops_for_quantised_models",
    
    # UNUSED PARAMS
    "sharding_strategy",
    "sharding_config_fsdp_config_use_orig_params",
    "sharding_config_fsdp_config_cpu_offload",
    "adaptive_batching",
    "adaptive_max_tokens",
    "query_rate",
    "total_input_tokens",
    "total_generated_tokens",
    
    # CONSTANT SETUP
    "date_time",
    "is_encoder_decoder",
    "task_type",
    "available_gpu_count",
    "gpu_model",
    "available_cpu_count",
    "cpu_model",
    "os",
    "python_version",
    "country",
    "region",
    "distributed_type",
    "decode_token_to_text",
    "inference_type",
    "backend",
    "total_params",
    "model_arch",

    # Validation:
    "max_input_tokens",
    "max_output_tokens",
    "number_input_prompts",
    
    # RESULTS
    # energy
    "total_energy_kwh",
    "total_energy_joules",
    # FLOPS and performance
    "flops",
    "tokens_per_joule",
    "joules_per_token",
    "flops_per_joule",
    "joules_per_flop",
    "total_inference_time_sec",
    "average_latency_ms_per_batch",
    "throughput_queries_per_sec",
    "throughput_tokens_per_sec",
    # CPU utilization
    "cpu_usage_percent",
    "cpu_memory_usage_bytes",
    # GPU utilization
    "gpu_utilization_percent_0", "gpu_utilization_percent_1", "gpu_utilization_percent_2", "gpu_utilization_percent_3",
    # Compute memory
    "gpu_current_memory_allocated_bytes",
    "gpu_max_memory_allocated_bytes",
    "gpu_current_memory_reserved_bytes",
    "gpu_max_memory_reserved_bytes",
    # Per-process metrics:
    "cpu_power_process_0", "cpu_power_process_1", "cpu_power_process_2", "cpu_power_process_3",
    "gpu_power_process_0", "gpu_power_process_1", "gpu_power_process_2", "gpu_power_process_3",
    "ram_power_process_0", "ram_power_process_1", "ram_power_process_2", "ram_power_process_3",
    "cpu_energy_process_0", "cpu_energy_process_1", "cpu_energy_process_2", "cpu_energy_process_3",
    "gpu_energy_process_0", "gpu_energy_process_1", "gpu_energy_process_2", "gpu_energy_process_3",
    "ram_energy_process_0", "ram_energy_process_1", "ram_energy_process_2", "ram_energy_process_3",
    "total_energy_kwh_process_0", "total_energy_kwh_process_1", "total_energy_kwh_process_2", "total_energy_kwh_process_3",
    "total_energy_joules_process_0", "total_energy_joules_process_1", "total_energy_joules_process_2", "total_energy_joules_process_3",
    # Global averages and totals:
    "cpu_power_avg",
    "gpu_power_avg",
    "ram_power_avg",
    "cpu_energy_total",
    "gpu_energy_total",
    "ram_energy_total",
    # per-process emissions
    "per-process_emissions_0", "per-process_emissions_1", "per-process_emissions_2", "per-process_emissions_3"
]

# Load CSV and clean it
def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """
    Loads a CSV from csv_path and cleans the column names and order.
    """
    df = pd.read_csv(csv_path)
    df_cleaned = clean_and_reorder_columns(df, desired_order)
    return df_cleaned

# -----------------------------------------------------------------
# Drop unused columns (two rounds)
# -----------------------------------------------------------------

def drop_unused_columns_1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop a first set of predefined columns that are not needed.
    """
    columns_to_drop_1 = [
        "sharding_strategy",
        "sharding_config_fsdp_config_use_orig_params",
        "sharding_config_fsdp_config_cpu_offload",
        "adaptive_batching",
        "adaptive_max_tokens",
        "query_rate",
        "is_encoder_decoder",
        "task_type",
        "available_gpu_count",
        "gpu_model",
        "available_cpu_count",
        "cpu_model",
        "os",
        "python_version",
        "country",
        "region",
        "distributed_type",
        "decode_token_to_text",
        "inference_type",
        "backend",
        "model_arch",
        "gpu_current_memory_allocated_bytes",
        "gpu_max_memory_allocated_bytes",
        "gpu_current_memory_reserved_bytes",
        "gpu_max_memory_reserved_bytes",
        "per-process_emissions_0", "per-process_emissions_1", "per-process_emissions_2", "per-process_emissions_3"
    ]
    return df.drop(columns=columns_to_drop_1, errors='ignore')

def drop_unused_columns_2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop a second set of predefined columns that are not needed.
    """
    columns_to_drop_2 = [
        "cpu_usage_percent",
        "cpu_memory_usage_bytes",
        #"gpu_utilization_percent_0",
        #"gpu_utilization_percent_1",
        #"gpu_utilization_percent_2",
        #"gpu_utilization_percent_3",
        "gpu_current_memory_allocated_bytes",
        "gpu_max_memory_allocated_bytes",
        "cpu_current_memory_allocated_bytes",
        "cpu_max_memory_allocated_bytes",
        "cpu_power_process_0",
        "cpu_power_process_1",
        "cpu_power_process_2",
        "cpu_power_process_3",
        #"gpu_power_process_0",
        #"gpu_power_process_1",
        #"gpu_power_process_2",
        #"gpu_power_process_3",
        "ram_power_process_0",
        "ram_power_process_1",
        "ram_power_process_2",
        "ram_power_process_3",
        "cpu_energy_process_0",
        "cpu_energy_process_1",
        "cpu_energy_process_2",
        "cpu_energy_process_3",
        #"gpu_energy_process_0",
        #"gpu_energy_process_1",
        #"gpu_energy_process_2",
        #"gpu_energy_process_3",
        #"ram_energy_process_0",
        #"ram_energy_process_1",
        #"ram_energy_process_2",
        #"ram_energy_process_3",
        "total_energy_joules_process_0",
        "total_energy_joules_process_1",
        "total_energy_joules_process_2",
        "total_energy_joules_process_3",
        "cpu_power_avg",
        "ram_energy_total",
        #"models"
    ]
    return df.drop(columns=columns_to_drop_2, errors='ignore')


# -----------------------------------------------------------------
# VERIFY, DIAGNOSE, CORRECT GENERATED TOKENS
# -----------------------------------------------------------------

def verify_generated_tokens(df: pd.DataFrame, total_generated_tokens_col='total_generated_tokens'):
    """
    Verify that the total_generated_tokens values are constant across runs.
    Prints detailed diagnostic info similar to filtering functions.
    """
    
    unique_tokens = df[total_generated_tokens_col].unique()
    original_counts = df[total_generated_tokens_col].value_counts().sort_index()
    
    if len(unique_tokens) == 1:
        print(f"✅ Total generated tokens value is constant: {unique_tokens[0]}")
        print(f"Original distribution:\n{original_counts}")
        print("--" * 50)
        return True
    else:
        unique_tokens_flag = False
        warnings.warn(f"⚠️ Total generated tokens values are NOT constant: {unique_tokens}", stacklevel=2)
        print(f"⚠️ Total generated tokens values are NOT constant: {unique_tokens}")
        print(f"Original distribution:\n{original_counts}")
        
        # Identify rows that do not have the dominant generated tokens value
        dominant_token = df[total_generated_tokens_col].mode().iloc[0]
        deviating_rows = df[df[total_generated_tokens_col] != dominant_token]
        deviating_configs = []
        if 'config_name' in df.columns:
            deviating_configs = deviating_rows['config_name'].unique().tolist()
        
        print(f"- Dominant token count: {dominant_token}")
        print(f"- Affected rows count: {len(deviating_rows)}")
        print(f"- Affected row indices: {deviating_rows.index.tolist()}")
        if deviating_configs:
            print(f"- Affected configs: {deviating_configs}")
        print("--" * 50)
        return False


def identify_total_generated_tokens_differentiators(df: pd.DataFrame, total_generated_tokens_col='total_generated_tokens', exclude_cols=None):
    """
    Identify columns that are constant within each total_generated_tokens group but differ between groups.
    """
    if exclude_cols is None:
        exclude_cols = []
    exclude_cols = set(exclude_cols + [total_generated_tokens_col])
    
    differentiators = {}
    unique_values = df[total_generated_tokens_col].unique()
    
    for col in df.columns:
        if col in exclude_cols:
            continue
        
        group_values = {}
        valid = True
        for val in unique_values:
            subset = df.loc[df[total_generated_tokens_col] == val, col]
            if isinstance(subset, pd.DataFrame):
                subset = subset.iloc[:, 0]  # ensure it's a Series
            values = subset.unique()
            if len(values) == 1:
                group_values[val] = values[0]
            else:
                valid = False
                break
        if valid and len(set(group_values.values())) > 1:
            differentiators[col] = group_values

    return differentiators

def filter_by_dominant_token_count(df: pd.DataFrame, token_col='total_generated_tokens') -> pd.DataFrame:
    original_counts = df[token_col].value_counts().sort_index()

    # If only one unique value, no need to filter
    if len(original_counts) == 1:
        print(f"✅ All rows have consistent '{token_col}' = {original_counts.index[0]}")
        print("--" * 50)
        return df

    # Otherwise, drop all rows not matching the most common value
    mode_val = df[token_col].mode().iloc[0]
    dropped_rows = df[df[token_col] != mode_val]
    dropped_row_config_names = dropped_rows['config_name'].unique()
    filtered_df = df[df[token_col] == mode_val]

    # Log warning and detailed info
    warnings.warn(f"⚠️ Dropped {len(dropped_rows)} rows due to inconsistent '{token_col}' values", stacklevel=2)
    print(f"⚠️ Dropped {len(dropped_rows)} rows due to inconsistent '{token_col}' values")
    print(f"- Filtering rows to dominant '{token_col}' = {mode_val}")
    print(f"- Retained {len(filtered_df)} of {len(df)} rows")
    print(f"- Dropped configs: {dropped_row_config_names.tolist()}")
    print(f"- Dropped row indices: {dropped_rows.index.tolist()}")
    print(f"Original distribution:\n{original_counts}")
    print("--" * 50)

    return filtered_df

# -----------------------------------------------------------------
# VERIFY, DIAGNOSE, CORRECT GENERATED FLOPS
# -----------------------------------------------------------------

def verify_flops(df: pd.DataFrame, flops_col='flops') -> None:
    """
    Verify that the FLOPs values are constant across runs.
    Prints detailed diagnostic info similar to filtering functions.
    """    
    unique_flops = df[flops_col].unique()
    original_counts = df[flops_col].value_counts().sort_index()
    
    if len(unique_flops) == 1:
        print(f"✅ FLOPs value is constant: {unique_flops[0]}")
        print(f"Original distribution:\n{original_counts}")
        print("--" * 50)
        return True
    else:
        unique_flops_flag = False
        warnings.warn(f"NB: FLOPs values are NOT constant: {unique_flops}", stacklevel=2)
        print(f"NB: FLOPs values are NOT constant: {unique_flops}")
        print(f"Original distribution:\n{original_counts}\n")
        
        # Identify rows that do not have the dominant FLOPs value
        dominant_flops = df[flops_col].mode().iloc[0]
        deviating_rows = df[df[flops_col] != dominant_flops]
        deviating_configs = []
        if 'config_name' in df.columns:
            deviating_configs = deviating_rows['config_name'].unique().tolist()
        
        print(f"Dominant FLOPs value: {dominant_flops}")
        print(f"- Affected rows count: {len(deviating_rows)}")
        print(f"- Affected row indices: {deviating_rows.index.tolist()}")
        if deviating_configs:
            print(f"- Affected configs: {deviating_configs}")
        print("--" * 50)
        return False
    

def identify_flop_differentiators(df: pd.DataFrame, flops_col='flops', exclude_cols=None):
    """
    Identify columns that are constant within each FLOP group but differ between groups.
    """
    if exclude_cols is None:
        exclude_cols = []
    exclude_cols = set(exclude_cols + [flops_col])
    
    differentiators = {}
    unique_flops = df[flops_col].unique()
    
    # Loop over each column in df excluding those in exclude_cols.
    for col in df.columns:
        if col in exclude_cols:
            continue
        
        group_values = {}
        valid = True
        for flop in unique_flops:
            subset = df.loc[df[flops_col] == flop, col]
            if isinstance(subset, pd.DataFrame):
                subset = subset.iloc[:, 0]  # ensure it's a Series
            values = subset.unique()
            if len(values) == 1:
                group_values[flop] = values[0]
            else:
                valid = False
                break
        if valid and len(set(group_values.values())) > 1:
            differentiators[col] = group_values
    
    return differentiators


def correct_flops(df: pd.DataFrame) -> pd.DataFrame:
    """
    Correct FLOPs based on model name suffix.
    """
    def compute(model_name, original):
        if model_name.endswith("1B"):
            return 16_949_970_993_152.0
        elif model_name.endswith("3B"):
            return 52_638_582_308_864.0
        elif model_name.endswith("8B"):
            raise ValueError("Please define correct FLOPs for 8B models.")
        else:
            return original
    
    df = df.copy()
    df['flops'] = df.apply(lambda r: compute(r['model'], r['flops']), axis=1)
    return df

# -----------------------------------------------------------------
# Create Derived Columns
# -----------------------------------------------------------------
def create_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['flops_per_token'] = df['flops'] / df['total_generated_tokens']
    df['energy_per_token_kwh'] = df['total_energy_kwh'] / df['total_generated_tokens']
    df['divergence_energy_flops'] = df['energy_per_token_kwh'] / df['flops_per_token']
    df['gpu_utilization_proc_all'] = df[['gpu_utilization_percent_0', 'gpu_utilization_percent_1', 'gpu_utilization_percent_2', 'gpu_utilization_percent_3']].mean(axis=1) / 4
    df['gpu_power_proc_all'] = df[['gpu_power_process_0', 'gpu_power_process_1', 'gpu_power_process_2', 'gpu_power_process_3']].mean(axis=1)

    return df

# -----------------------------------------------------------------
# aggregate config grouping metrics across runs
# -----------------------------------------------------------------

def add_config_group_stats(
    df: pd.DataFrame,
    group_by: Union[str, Sequence[str]] = 'config_name',
    cols: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """
    For each column in `cols`, adds two new columns:
      • {col}_mean  = group-by mean of col
      • {col}_std   = group-by std  of col
    Keeps all original rows.
    
    """
    df_out = df.copy()
    
    if isinstance(group_by, str):
        group_by = [group_by]
    
    if cols is None:
        cols = [
            'total_energy_kwh',
            'total_inference_time_sec',
            'average_latency_ms_per_batch',
            'throughput_queries_per_sec',
            'throughput_tokens_per_sec',
            'cpu_energy_total',
            'gpu_energy_total',
            'flops',
            'flops_per_token',
            'energy_per_token_kwh',
            'divergence_energy_flops'
        ]
    
    # sanity check
    missing = [c for c in group_by + cols if c not in df_out.columns]
    if missing:
        raise KeyError(f"These columns are missing: {missing}")
    
    # compute transform for each metric
    for c in cols:
        df_out[f"{c}_mean"] = df_out.groupby(group_by)[c].transform('mean')
        df_out[f"{c}_std"]  = df_out.groupby(group_by)[c].transform('std')
    
    return df_out

# -----------------------------------------------------------------
# 8. wrap pipeline
# -----------------------------------------------------------------

def run_load_clean_diagnose_data(csv_path: str = "results/controlled_results.csv") -> pd.DataFrame:
    # Load and clean data
    df = load_and_clean_data(csv_path)

    # Drop unused columns
    df = drop_unused_columns_1(df)
    df = drop_unused_columns_2(df)
    
    # sanity check: verify generated tokens
    if not verify_generated_tokens(df):
        # 1 identify differentiators 
        token_diff = identify_total_generated_tokens_differentiators(df)
        print("Total Generated Tokens Differentiators:")
        for col, vals in token_diff.items():
            print(f"{col}: {vals}")
    
        # 2 correct
        df = filter_by_dominant_token_count(df)
    
    # sanity check: verify FLOPs
    print("Round 1: Verfifying FLOPs on raw df")
    if not verify_flops(df):
        # 1 identify differentiators for FLOPs
        flop_diff = identify_flop_differentiators(df)
        print("FLOP Differentiators:")
        for col, vals in flop_diff.items():
            print(f"{col}: {vals}")
    
        # 2 correct FLOPs if necessary
        df = correct_flops(df)
        print("Round 2: Verfifying FLOPs on corrected df")
        verify_flops(df)
        
    if len(df['flops'].unique()) == len(df['model'].unique()):
        print("✅ FLOPs are unique per model")
    else:
        print("⚠️ FLOPs are NOT unique per model")
    
    print("--" * 50)
    # check cycles
    print(f"cyles present: {df['cycle_id'].unique()}")
    
    # create derived stats
    df = create_derived_columns(df)
    
    # aggregate
    df = add_config_group_stats(df)
    
    # clean up
    df['model'] = df['model'].str.removeprefix('meta-llama/')
    
    return df


# -----------------------------------------------------------------
# 9. Main execution block
# -----------------------------------------------------------------
if __name__ == "__main__":
    df = run_load_clean_diagnose_data()
